"""Reproducible prototype for CO3057 Project 2.

Input images are supplied by the user in ``assets/``. The feature methods
are educational NumPy/SciPy implementations: Harris corners with normalized
local-patch descriptors, and FAST corners with oriented BRIEF descriptors
(an ORB-style binary method).
"""
from __future__ import annotations
import json, math, re, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def save_rgb(arr, path):
    Image.fromarray(np.asarray(np.clip(arr, 0, 255), dtype=np.uint8), 'RGB').save(path)


def image_files():
    if not ASSETS.is_dir():
        raise FileNotFoundError(
            f'Create {ASSETS} and upload the input images there before running the script.'
        )
    paths = sorted(
        path for path in ASSETS.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f'No supported images found in {ASSETS}.')
    return paths


def role_match(path, aliases):
    tokens = set(re.split(r'[^a-z0-9]+', path.stem.lower()))
    return bool(tokens.intersection(aliases))


def find_role(paths, aliases, label):
    matches = [path for path in paths if role_match(path, aliases)]
    if len(matches) > 1:
        names = ', '.join(path.name for path in matches)
        raise ValueError(f'Use only one {label} image in {ASSETS}; found {names}.')
    return matches[0] if matches else None


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def load_source_and_mask(paths):
    source_path = find_role(paths, {'source', 'foreground', 'object', 'src'}, 'source')
    background_path = find_role(
        paths, {'background', 'destination', 'target', 'dest', 'bg'}, 'background'
    )
    mask_paths = [path for path in paths if path not in {source_path, background_path}]
    mask_path = find_role(mask_paths, {'mask', 'silhouette', 'alpha'}, 'mask')
    if source_path is None or background_path is None:
        raise ValueError(
            f'Upload images named source and background to {ASSETS} '
            '(for example source.jpg and background.jpg).'
        )

    with Image.open(source_path) as image:
        rgba = np.asarray(image.convert('RGBA'), dtype=np.uint8)
    source = rgba[:, :, :3]
    if mask_path is not None:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert('L'), dtype=np.uint8)
    elif np.any(rgba[:, :, 3] < 255):
        mask = rgba[:, :, 3]
    else:
        raise ValueError(
            f'Upload a mask image for {source_path.name}, or use a source PNG with transparency.'
        )

    destination = load_rgb(background_path)
    if mask.shape != source.shape[:2]:
        raise ValueError('The source image and mask must have identical dimensions.')
    if source.shape[0] > destination.shape[0] or source.shape[1] > destination.shape[1]:
        raise ValueError('The source image must fit inside the background image.')
    if not np.any(mask):
        raise ValueError('The uploaded mask must contain at least one non-zero pixel.')
    offset = (
        (destination.shape[1] - source.shape[1]) // 2,
        (destination.shape[0] - source.shape[0]) // 2,
    )
    return source, mask, destination, offset, [source_path, background_path, mask_path]


def load_panorama_views(paths, excluded_paths):
    excluded = {path for path in excluded_paths if path is not None}
    view_names = [
        path for path in paths
        if path not in excluded and role_match(path, {'view', 'frame'})
    ]
    panorama_names = [
        path for path in paths
        if path not in excluded and role_match(path, {'panorama'})
        and re.search(r'\d', path.stem)
    ]
    named_views = view_names if len(view_names) >= 2 else panorama_names
    view_paths = named_views if len(named_views) >= 2 else [path for path in paths if path not in excluded]
    if len(view_paths) < 2:
        raise ValueError(
            f'Upload at least two panorama images to {ASSETS}; '
            'name them view_1, view_2, and so on when other input images are present.'
        )
    views = [load_rgb(path) for path in view_paths]
    shape = views[0].shape
    if any(view.shape != shape for view in views[1:]):
        raise ValueError('All panorama images must have identical dimensions.')
    return views, view_paths


def poisson_clone(source, mask, target, offset):
    sx,sy=offset; h,w=mask.shape
    assert sy+h<=target.shape[0] and sx+w<=target.shape[1]
    # Active mask only. Use 4-neighbour Poisson equations with target Dirichlet boundary.
    yy,xx=np.nonzero(mask>0)
    n=len(xx); index=-np.ones((h,w),dtype=np.int32); index[yy,xx]=np.arange(n)
    rows=[]; cols=[]; vals=[]
    src=source.astype(np.float64)
    dst=target[sy:sy+h,sx:sx+w].astype(np.float64)
    rhs=np.zeros((n,3),dtype=np.float64)
    neigh=[(0,1),(0,-1),(1,0),(-1,0)]
    for k,(y,x) in enumerate(zip(yy,xx)):
        rows.append(k); cols.append(k); vals.append(4.0)
        for dy,dx in neigh:
            ny,nx=y+dy,x+dx
            if ny<0 or ny>=h or nx<0 or nx>=w: continue
            rhs[k]+=src[y,x]-src[ny,nx]
            j=index[ny,nx]
            if j>=0:
                rows.append(k); cols.append(int(j)); vals.append(-1.0)
            else:
                rhs[k]+=dst[ny,nx]
    A=sparse.csr_matrix((vals,(rows,cols)),shape=(n,n))
    sol=np.stack([spsolve(A,rhs[:,c]) for c in range(3)],axis=1)
    result=target.copy().astype(np.float64)
    patch=result[sy:sy+h,sx:sx+w]
    patch[yy,xx]=sol
    return np.uint8(np.clip(result,0,255))


def direct_clone(source,mask,target,offset):
    sx,sy=offset
    out=target.copy()
    h,w=mask.shape
    patch=out[sy:sy+h,sx:sx+w]
    patch[mask>0]=source[mask>0]
    return out


def seam_edge_mean(image,mask,offset):
    sx,sy=offset; h,w=mask.shape
    patch=image[sy:sy+h,sx:sx+w].astype(float)
    m=mask>0
    values=[]
    for dy,dx in [(0,1),(1,0)]:
        a=m[:h-dy or None,:w-dx or None]
        b=m[dy:,dx:]
        cross=a^b
        if np.any(cross):
            p=patch[:h-dy or None,:w-dx or None][cross]
            q=patch[dy:,dx:][cross]
            values.extend(np.mean(np.abs(p-q),axis=1).tolist())
    return float(np.mean(values)) if values else float('nan')


def harris_ncc_features(gray, max_points=450):
    g=gray.astype(np.float32)/255.0
    gx=ndimage.sobel(g,axis=1,mode='reflect')/8.0
    gy=ndimage.sobel(g,axis=0,mode='reflect')/8.0
    a=ndimage.gaussian_filter(gx*gx,1.4)
    b=ndimage.gaussian_filter(gx*gy,1.4)
    c=ndimage.gaussian_filter(gy*gy,1.4)
    response=a*c-b*b-0.045*(a+c)**2
    local=ndimage.maximum_filter(response,size=5,mode='nearest')
    peaks=(response>=local-1e-10)&(response>max(float(response.max())*0.0008,1e-7))
    candidates=[]
    for y,x in np.argwhere(peaks):
        if y<12 or y>=g.shape[0]-12 or x<12 or x>=g.shape[1]-12: continue
        candidates.append((float(response[y,x]),int(x),int(y),1.6))
    candidates.sort(reverse=True)
    # Spatial suppression avoids descriptors clustered at one corner.
    chosen=[]; cells=set()
    for score,x,y,sigma in candidates:
        cell=(x//7,y//7)
        if cell in cells: continue
        cells.add(cell); chosen.append((x,y,sigma))
        if len(chosen)>=max_points: break
    if not chosen: return np.empty((0,2)),np.empty((0,81),np.float32)
    coords=np.linspace(-6,6,9,dtype=np.float32)
    xx,yy=np.meshgrid(coords,coords)
    desc=[]; pts=[]
    for x,y,sigma in chosen:
        # Mean-center and L2-normalize a 9x9 intensity patch (NCC-style).
        v=ndimage.map_coordinates(g,[y+yy,x+xx],order=1,mode='reflect').ravel()
        v=v-v.mean(); v=v/(np.linalg.norm(v)+1e-8)
        desc.append(v); pts.append((x,y))
    return np.asarray(pts,float),np.asarray(desc,np.float32)


FAST_RING=[(-3,-1),(-3,0),(-3,1),(-2,2),(-1,3),(0,3),(1,3),(2,2),(3,1),(3,0),(3,-1),(2,-2),(1,-3),(0,-3),(-1,-3),(-2,-2)]

def orb_style_features(gray,max_points=600,seed=91):
    g=gray.astype(np.float32)/255.0
    h,w=g.shape; center=g[3:h-3,3:w-3]
    ring=np.stack([g[3+dy:h-3+dy,3+dx:w-3+dx] for dy,dx in FAST_RING])
    delta=ring-center[None,:,:]
    thr=0.075
    bright=delta>thr; dark=delta<-thr
    corner=np.zeros(center.shape,bool)
    # FAST-9: at least nine consecutive pixels brighter or darker than center.
    for start in range(16):
        inds=[(start+j)%16 for j in range(9)]
        corner |= bright[inds].all(axis=0) | dark[inds].all(axis=0)
    score=np.maximum(np.abs(delta)-thr,0).sum(axis=0)
    peaks=score==ndimage.maximum_filter(score,size=3,mode='nearest')
    ys,xs=np.nonzero(corner&peaks)
    vals=score[ys,xs]
    order=np.argsort(vals)[::-1]
    points=[]
    used=set()
    for idx in order:
        y=int(ys[idx]+3); x=int(xs[idx]+3)
        cell=(x//6,y//6)
        if cell in used: continue
        used.add(cell); points.append((x,y))
        if len(points)>=max_points: break
    if not points: return np.empty((0,2)),np.empty((0,32),np.uint8)
    rng=np.random.default_rng(seed)
    pairs=rng.integers(-8,9,size=(256,4))
    pts=[]; descriptors=[]
    for x,y in points:
        y0=max(0,y-7); y1=min(h,y+8); x0=max(0,x-7); x1=min(w,x+8)
        patch=g[y0:y1,x0:x1]
        py,px=np.mgrid[y0:y1,x0:x1]
        m10=float(np.sum((px-x)*patch)); m01=float(np.sum((py-y)*patch))
        ori=math.atan2(m01,m10)
        c,s=math.cos(ori),math.sin(ori)
        ax=c*pairs[:,0]-s*pairs[:,1]+x; ay=s*pairs[:,0]+c*pairs[:,1]+y
        bx=c*pairs[:,2]-s*pairs[:,3]+x; by=s*pairs[:,2]+c*pairs[:,3]+y
        av=ndimage.map_coordinates(g,[ay,ax],order=1,mode='nearest')
        bv=ndimage.map_coordinates(g,[by,bx],order=1,mode='nearest')
        bit=av<bv
        descriptors.append(np.packbits(bit.astype(np.uint8)))
        pts.append((x,y))
    return np.asarray(pts,float),np.asarray(descriptors,np.uint8)


def match_descriptors(desc_a,desc_b,kind,ratio=0.78):
    if len(desc_a)<2 or len(desc_b)<2: return np.empty((0,2),int)
    if kind=='float':
        aa=(desc_a**2).sum(axis=1)[:,None]
        bb=(desc_b**2).sum(axis=1)[None,:]
        d2=np.maximum(aa+bb-2*desc_a@desc_b.T,0)
        d=np.sqrt(d2)
    else:
        lut=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
        xor=desc_a[:,None,:]^desc_b[None,:,:]
        d=lut[xor].sum(axis=2).astype(float)
    nearest=np.argpartition(d,1,axis=1)[:,:2]
    reverse_best=np.argmin(d,axis=0)
    out=[]
    for i,(j0,j1) in enumerate(nearest):
        if d[i,j0]>d[i,j1]: j0,j1=j1,j0
        if reverse_best[j0]==i and d[i,j0] < ratio*max(d[i,j1],1e-8): out.append((i,int(j0)))
    return np.asarray(out,dtype=int).reshape(-1,2)


def normalize_points(pts):
    c=pts.mean(axis=0); dist=np.mean(np.linalg.norm(pts-c,axis=1))+1e-9
    s=math.sqrt(2)/dist
    T=np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1]],float)
    q=(T@np.column_stack([pts,np.ones(len(pts))]).T).T
    return q[:,:2],T


def fit_homography(src,dst):
    if len(src)<4: return None
    a,Ts=normalize_points(src); b,Td=normalize_points(dst)
    A=[]
    for (x,y),(u,v) in zip(a,b):
        A.append([-x,-y,-1,0,0,0,u*x,u*y,u])
        A.append([0,0,0,-x,-y,-1,v*x,v*y,v])
    A=np.asarray(A,float)
    _,_,vh=np.linalg.svd(A,full_matrices=True)
    Hn=vh[-1].reshape(3,3)
    H=np.linalg.inv(Td)@Hn@Ts
    if abs(H[2,2])>1e-12: H/=H[2,2]
    return H


def transform_points(H,pts):
    q=(H@np.column_stack([pts,np.ones(len(pts))]).T).T
    z=q[:,2:3]
    return q[:,:2]/np.where(np.abs(z)<1e-9,np.nan,z)


def ransac_homography(src,dst,threshold=4.0,iterations=1000,seed=32):
    if len(src)<4: return None,np.zeros(len(src),bool),float('inf')
    rng=np.random.default_rng(seed)
    best=np.zeros(len(src),bool); best_rmse=float('inf')
    for _ in range(iterations):
        ids=rng.choice(len(src),4,replace=False)
        H=fit_homography(src[ids],dst[ids])
        if H is None or not np.all(np.isfinite(H)): continue
        err=np.linalg.norm(transform_points(H,src)-dst,axis=1)
        inl=err<threshold
        n=int(inl.sum())
        if n>=4:
            rmse=float(np.sqrt(np.mean(err[inl]**2)))
            if n>best.sum() or (n==best.sum() and rmse<best_rmse):
                best=inl;best_rmse=rmse
    if best.sum()>=4:
        H=fit_homography(src[best],dst[best])
        err=np.linalg.norm(transform_points(H,src)-dst,axis=1)
        best=err<threshold
        if best.sum()>=4:
            H=fit_homography(src[best],dst[best])
            best_rmse=float(np.sqrt(np.mean(np.linalg.norm(transform_points(H,src[best])-dst[best],axis=1)**2)))
        return H,best,best_rmse
    return None,best,best_rmse


def stitch_views(views, transforms_to_base):
    h,w=views[0].shape[:2]
    corners=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],float)
    allc=np.concatenate([transform_points(T,corners) for T in transforms_to_base],axis=0)
    mn=np.floor(np.nanmin(allc,axis=0)).astype(int); mx=np.ceil(np.nanmax(allc,axis=0)).astype(int)
    shift=np.array([[1,0,-mn[0]],[0,1,-mn[1]],[0,0,1]],float)
    oh,ow=mx[1]-mn[1]+1,mx[0]-mn[0]+1
    accum=np.zeros((oh,ow,3),float); weight=np.zeros((oh,ow),float)
    yy,xx=np.mgrid[0:oh,0:ow]
    for view,T in zip(views,transforms_to_base):
        H=shift@T
        p=np.linalg.inv(H)@np.stack([xx.ravel(),yy.ravel(),np.ones(oh*ow)])
        p/=p[2:3,:]
        sx=p[0].reshape(oh,ow); sy=p[1].reshape(oh,ow)
        valid=(sx>=0)&(sx<=w-1)&(sy>=0)&(sy<=h-1)
        warped=np.stack([ndimage.map_coordinates(view[:,:,c].astype(float),[sy,sx],order=1,mode='constant',cval=0) for c in range(3)],axis=2)
        vm=valid.astype(np.uint8)
        dist=ndimage.distance_transform_edt(vm)
        wt=np.where(valid,np.maximum(dist,1.0),0)
        accum+=warped*wt[:,:,None]; weight+=wt
    pano=accum/np.maximum(weight[:,:,None],1e-8)
    return np.uint8(np.clip(pano,0,255)), (int(oh),int(ow))


def gray(rgb):
    a=rgb.astype(np.float32)
    return 0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2]


def run_feature_method(views,method):
    gray_views=[gray(v) for v in views]
    extractor=harris_ncc_features if method=='Harris-NCC' else orb_style_features
    desc_kind='float' if method=='Harris-NCC' else 'binary'
    results=[]; transforms=[np.eye(3)]
    t0=time.perf_counter()
    features=[extractor(g) for g in gray_views]
    extraction_time=time.perf_counter()-t0
    pair_matches=[]
    for i in range(len(views)-1):
        p1,d1=features[i]; p2,d2=features[i+1]
        matches=match_descriptors(d1,d2,desc_kind,ratio=0.70 if desc_kind=='float' else 0.88)
        src=p1[matches[:,0]] if len(matches) else np.empty((0,2))
        dst=p2[matches[:,1]] if len(matches) else np.empty((0,2))
        H,inliers,rmse=ransac_homography(src,dst,threshold=4.0,iterations=850,seed=70+i)
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-10:
            # Identity is only a fail-safe for rendering; metrics retain the failure.
            H=np.eye(3)
        results.append({'pair':f'{i+1}-{i+2}','keypoints_a':int(len(p1)),'keypoints_b':int(len(p2)),
                        'tentative_matches':int(len(matches)),'inliers':int(inliers.sum()),
                        'inlier_ratio':float(inliers.mean()) if len(inliers) else 0.0,
                        'inlier_rmse_px':float(rmse)})
        pair_matches.append(H)
    # Matrices mapping each image into view 1 coordinates.
    for H in pair_matches:
        transforms.append(transforms[-1]@np.linalg.inv(H))
    pano,shape=stitch_views(views,transforms)
    save_rgb(pano,RESULTS/f'panorama_{method.lower().replace("-","_")}.png')
    return {'method':method,'extraction_time_s':float(extraction_time),'pairs':results,'panorama_shape':shape},pano,features


def plot_composites(target,direct,poisson,mask,offset):
    fig,axs=plt.subplots(1,4,figsize=(15,4.2))
    items=[(target,'Background'),(np.dstack([mask]*3),'Manual object mask'),(direct,'Direct replacement'),(poisson,'Poisson gradient-domain')]
    for ax,(im,title) in zip(axs,items):
        ax.imshow(im);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('Gradient-domain editing from uploaded images',fontsize=14)
    fig.tight_layout();fig.savefig(RESULTS/'gradient_editing_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    crop=(slice(offset[1]-10,offset[1]+mask.shape[0]+10),slice(offset[0]-10,offset[0]+mask.shape[1]+10))
    fig,axs=plt.subplots(1,3,figsize=(11,4))
    for ax,im,title in zip(axs,[target,direct,poisson],['Destination','Direct','Poisson']):
        ax.imshow(im[crop]);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(RESULTS/'gradient_seam_detail.png',dpi=180,bbox_inches='tight');plt.close(fig)


def plot_panorama(views,pano_sift,pano_orb):
    fig=plt.figure(figsize=(14,8))
    columns=max(2*len(views),4)
    grid=fig.add_gridspec(2,columns,height_ratios=[1,1.15],hspace=.22,wspace=.08)
    for i,view in enumerate(views):
        ax=fig.add_subplot(grid[0,2*i:2*i+2]);ax.imshow(view);ax.set_title(f'View {i+1}');ax.axis('off')
    middle=columns//2
    ax=fig.add_subplot(grid[1,0:middle]);ax.imshow(pano_sift);ax.set_title('Harris-NCC panorama');ax.axis('off')
    ax=fig.add_subplot(grid[1,middle:]);ax.imshow(pano_orb);ax.set_title('ORB-style panorama');ax.axis('off')
    fig.suptitle('Feature-based stitching from uploaded images',fontsize=14)
    fig.savefig(RESULTS/'panorama_results.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    # Part 1: direct baseline and discrete Poisson solve.
    paths=image_files()
    source,mask,dest,offset,composite_paths=load_source_and_mask(paths)
    direct=direct_clone(source,mask,dest,offset)
    t0=time.perf_counter();poisson=poisson_clone(source,mask,dest,offset);poisson_time=time.perf_counter()-t0
    save_rgb(direct,RESULTS/'gradient_direct.png');save_rgb(poisson,RESULTS/'gradient_poisson.png')
    metrics={'gradient_domain':{'mask_pixels':int((mask>0).sum()),'direct_boundary_edge_mean_l1':seam_edge_mean(direct,mask,offset),
      'poisson_boundary_edge_mean_l1':seam_edge_mean(poisson,mask,offset),'poisson_runtime_s':poisson_time,
      'input_files':[path.name for path in composite_paths if path is not None]}}
    plot_composites(dest,direct,poisson,mask,offset)
    # Part 2: one common pipeline for two descriptor families.
    views,view_paths=load_panorama_views(paths,composite_paths)
    sift,ps,_=run_feature_method(views,'Harris-NCC')
    orb,po,_=run_feature_method(views,'ORB-style')
    metrics['stitching']={'view_count':len(views),'view_shape':list(views[0].shape[:2]),
       'input_files':[path.name for path in view_paths],
       'methods':[sift,orb]}
    plot_panorama(views,ps,po)
    (RESULTS/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
