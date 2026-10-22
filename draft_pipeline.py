"""Reproducible prototype for CO3057 Project 2.

All images are procedurally generated and are explicitly marked synthetic.
The feature methods are educational NumPy/SciPy implementations: Harris
corners with normalized local-patch descriptors, and FAST corners with
oriented BRIEF descriptors (an ORB-style binary method).
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
RESULTS = ROOT / 'results'
ASSETS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
RNG = np.random.default_rng(407)


def save_rgb(arr, path):
    Image.fromarray(np.asarray(np.clip(arr, 0, 255), dtype=np.uint8), 'RGB').save(path)


def gradient_background(w, h, top=(118, 181, 220), bottom=(218, 225, 220), seed=1):
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    a = np.array(top, dtype=np.float32)[None, None, :]
    b = np.array(bottom, dtype=np.float32)[None, None, :]
    out = a * (1-y) + b*y
    out = np.broadcast_to(out, (h, w, 3)).copy()
    noise = rng.normal(0, 2.2, (h, w, 1)).astype(np.float32)
    out += noise
    return np.uint8(np.clip(out, 0, 255))


def make_panorama_scene():
    w, h = 1680, 520
    arr = gradient_background(w, h, (115, 176, 213), (224, 222, 192), 23)
    im = Image.fromarray(arr, 'RGB')
    d = ImageDraw.Draw(im, 'RGB')
    # Distant hills and a textured foreground establish broad, low-frequency structure.
    d.polygon([(0, 238),(190,190),(370,235),(560,182),(760,234),(980,180),(1180,230),(1390,178),(1680,228),(1680,520),(0,520)], fill=(93,126,99))
    d.rectangle((0, 294, w, 520), fill=(165,150,119))
    d.polygon([(0,355),(500,318),(1080,360),(1680,315),(1680,520),(0,520)], fill=(89,91,83))
    d.line((0,355,w,315), fill=(223,211,177), width=9)
    # Three connected building facades with dense, varied corners.
    blocks = [(55,114,385,305),(416,86,772,306),(812,126,1145,310),(1182,92,1609,303)]
    facade_cols=[(202,174,138),(173,185,180),(188,154,138),(173,171,151)]
    for bi, box in enumerate(blocks):
        x0,y0,x1,y1=box
        col=facade_cols[bi]
        d.rectangle(box, fill=col, outline=(55,58,56), width=5)
        d.rectangle((x0,y0,x1,y0+21), fill=(75,78,72))
        # Horizontal masonry bands.
        for yy in range(y0+32,y1,45):
            d.line((x0,yy,x1,yy+((bi%2)*2)), fill=(119,111,98), width=2)
        wx=42 if bi%2==0 else 36
        for yy in range(y0+40,y1-35,54):
            for xx in range(x0+23,x1-20,wx):
                ww=22 if (xx//wx)%2 else 25
                hh=25 if (yy//54)%2 else 31
                d.rectangle((xx,yy,xx+ww,yy+hh), fill=(42,68,78), outline=(237,219,177), width=3)
                d.line((xx+ww//2,yy,xx+ww//2,yy+hh), fill=(219,209,184), width=2)
                if (xx+yy+bi)%3==0:
                    d.rectangle((xx+4,yy+5,xx+9,yy+10), fill=(255,223,126))
        d.rectangle((x0+10,y1-41,x1-10,y1-30), fill=(122,92,68))
        if bi in (1,3):
            d.rectangle((x0+120,y1-79,x0+186,y1), fill=(88,69,54), outline=(219,206,175), width=3)
    # Trees and lamps give non-repeating features across the overlap.
    for x,y,r in [(22,274,46),(392,269,42),(787,275,52),(1160,268,45),(1640,276,48)]:
        d.rectangle((x-6,y,x+7,370), fill=(81,63,45))
        for dx,dy,rr in [(-21,-14,r),(20,-18,int(r*.85)),(0,-43,int(r*.8)),(-7,13,int(r*.7))]:
            d.ellipse((x+dx-rr,y+dy-rr,x+dx+rr,y+dy+rr), fill=(49,105,73), outline=(38,82,62), width=2)
    for x in (320, 710, 1070, 1480):
        d.line((x,245,x,360), fill=(38,43,43), width=5)
        d.line((x,245,x+28,236), fill=(38,43,43), width=4)
        d.ellipse((x+20,229,x+38,242), fill=(252,231,169), outline=(50,50,43), width=2)
    # Pavement seams, small signs, and deterministic texture patches.
    for x in range(-30,w,52):
        d.line((x,385,x+25,520), fill=(130,122,105), width=2)
    for y in range(398,520,31):
        d.line((0,y,w,y), fill=(126,118,103), width=2)
    for x in (140, 545, 905, 1350):
        d.rectangle((x,216,x+66,255), fill=(53,83,89), outline=(238,223,191), width=3)
        d.text((x+8,228), ['CAFE','MARKET','HOTEL','BOOKS'][[140,545,905,1350].index(x)], fill=(245,236,203))
    # Fine, deterministic noise makes the scene less vector-like and supplies texture.
    out=np.asarray(im).astype(np.float32)
    noise=np.random.default_rng(24).normal(0,2.0,(h,w,1))
    out=np.clip(out+noise,0,255).astype(np.uint8)
    save_rgb(out, ASSETS/'panorama_master_synthetic.png')
    return out


def projective_about_center(w,h, angle_deg=0.0, scale=1.0, shear=0.0, px=0.0, py=0.0):
    t1=np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]],float)
    a=math.radians(angle_deg)
    m=np.array([[scale*math.cos(a), -math.sin(a)+shear, 0],
                [math.sin(a), scale*math.cos(a), 0],
                [px,py,1]],float)
    t2=np.array([[1,0,w/2],[0,1,h/2],[0,0,1]],float)
    return t2@m@t1


def warp_from_scene(scene, H_scene_to_view, out_shape):
    h,w=out_shape
    yy,xx=np.mgrid[0:h,0:w]
    q=np.stack([xx.ravel(),yy.ravel(),np.ones(h*w)])
    scene_p=np.linalg.inv(H_scene_to_view)@q
    scene_p/=scene_p[2:3,:]
    sx=scene_p[0].reshape(h,w); sy=scene_p[1].reshape(h,w)
    vals=[]
    for c in range(3):
        ch=ndimage.map_coordinates(scene[:,:,c].astype(np.float32),[sy,sx],order=1,mode='constant',cval=0)
        vals.append(ch)
    return np.stack(vals,axis=2)


def generate_panorama_views(scene):
    h,w=480,720
    offsets=[0,470,940]
    params=[(0.0,1.0,0.0,0.0,0.0),(1.2,1.003,0.002,0.000003,-0.000002),(-1.0,0.997,-0.001,-0.000002,0.000003)]
    views=[]; hs=[]
    for i,(xoff,p) in enumerate(zip(offsets,params)):
        P=projective_about_center(w,h,*p)
        C=np.array([[1,0,-xoff],[0,1,0],[0,0,1]],float)
        H=P@C
        hs.append(H)
        v=warp_from_scene(scene,H,(h,w))
        save_rgb(v,ASSETS/f'panorama_view_{i+1}_synthetic.png')
        views.append(np.uint8(np.clip(v,0,255)))
    return views,hs


def make_composite_pair():
    sw,sh=512,384
    source=gradient_background(sw,sh,(91,166,211),(194,218,223),31)
    src_im=Image.fromarray(source,'RGB')
    sd=ImageDraw.Draw(src_im,'RGB')
    # Light cloud bands for source context.
    for cx,cy,rx in [(80,90,54),(405,84,72),(110,270,48),(425,286,58)]:
        for dx,dy,rr in [(-20,0,rx*.55),(8,-12,rx*.65),(31,1,rx*.5)]:
            sd.ellipse((cx+dx-rr,cy+dy-rr*.38,cx+dx+rr,cy+dy+rr*.38),fill=(220,235,237))
    # Balloon and basket; object mask is a manual silhouette.
    mask=Image.new('L',(sw,sh),0); md=ImageDraw.Draw(mask)
    # Basket ropes first so the single object silhouette includes them.
    balloon_box=(187,42,326,221)
    sd.line((213,198,228,273),fill=(69,54,42),width=4)
    sd.line((300,198,285,273),fill=(69,54,42),width=4)
    sd.polygon([(221,267),(292,267),(284,308),(230,308)],fill=(132,82,49),outline=(64,48,36))
    sd.line((229,278,285,278),fill=(224,178,102),width=4)
    # A colorful paneled envelope.
    sd.ellipse(balloon_box,fill=(203,62,43),outline=(85,45,39),width=5)
    sd.pieslice(balloon_box,180,360,fill=(232,153,49),outline=(85,45,39),width=3)
    sd.polygon([(187,131),(257,48),(257,219),(204,221)],fill=(222,83,48))
    sd.polygon([(257,48),(326,131),(311,216),(257,219)],fill=(239,183,78))
    sd.line((257,48,257,220),fill=(89,57,42),width=3)
    sd.line((191,144,323,144),fill=(103,58,40),width=2)
    sd.ellipse((236,300,277,318),fill=(48,47,40))
    # Mask made from the envelope, ropes, basket, and burner silhouette.
    md.ellipse(balloon_box,fill=255)
    md.line((213,198,228,273),fill=255,width=7)
    md.line((300,198,285,273),fill=255,width=7)
    md.polygon([(217,263),(296,263),(291,313),(224,313)],fill=255)
    md.ellipse((231,295,282,322),fill=255)
    source=np.asarray(src_im)
    mask=np.asarray(mask)

    dw,dh=760,480
    dest=gradient_background(dw,dh,(137,190,220),(203,218,211),47)
    dest_im=Image.fromarray(dest,'RGB'); d=ImageDraw.Draw(dest_im,'RGB')
    # Layered hills with fine grass stripes.
    d.polygon([(0,286),(180,230),(350,285),(520,222),(760,278),(760,480),(0,480)],fill=(92,135,98))
    d.polygon([(0,335),(160,299),(335,337),(550,286),(760,335),(760,480),(0,480)],fill=(69,112,77))
    for x in range(0,dw,17):
        y=350+int(18*math.sin(x/62))
        d.line((x,y,x+10,y-10),fill=(132,154,92),width=2)
    for cx,cy,r in [(94,273,36),(630,255,46),(700,285,31),(350,302,23)]:
        d.rectangle((cx-5,cy,cx+5,cy+82),fill=(79,61,43))
        for dx,dy,rr in [(-15,-7,r*.7),(14,-9,r*.65),(0,-25,r*.8)]:
            d.ellipse((cx+dx-rr,cy+dy-rr,cx+dx+rr,cy+dy+rr),fill=(48,100,65))
    dest=np.asarray(dest_im)
    # Place the source object in the upper-right sky region.
    x0,y0=220,42
    return source,mask,dest,(x0,y0)


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


def run_feature_method(views,Hs,method):
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
        Hgt=Hs[i+1]@np.linalg.inv(Hs[i])
        src=p1[matches[:,0]] if len(matches) else np.empty((0,2))
        dst=p2[matches[:,1]] if len(matches) else np.empty((0,2))
        H,inliers,rmse=ransac_homography(src,dst,threshold=4.0,iterations=850,seed=70+i)
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-10:
            # Identity is only a fail-safe for rendering; metrics retain the failure.
            H=np.eye(3)
        true_pts=np.array([[0,0],[719,0],[719,479],[0,479],[360,240]],float)
        geom=float(np.sqrt(np.mean(np.sum((transform_points(H,true_pts)-transform_points(Hgt,true_pts))**2,axis=1))))
        results.append({'pair':f'{i+1}-{i+2}','keypoints_a':int(len(p1)),'keypoints_b':int(len(p2)),
                        'tentative_matches':int(len(matches)),'inliers':int(inliers.sum()),
                        'inlier_ratio':float(inliers.mean()) if len(inliers) else 0.0,
                        'inlier_rmse_px':float(rmse),'homography_corner_rmse_px':geom})
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
    fig.suptitle('Gradient-domain editing (synthetic demonstration)',fontsize=14)
    fig.tight_layout();fig.savefig(RESULTS/'gradient_editing_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    crop=(slice(offset[1]-10,offset[1]+mask.shape[0]+10),slice(offset[0]-10,offset[0]+mask.shape[1]+10))
    fig,axs=plt.subplots(1,3,figsize=(11,4))
    for ax,im,title in zip(axs,[target,direct,poisson],['Destination','Direct','Poisson']):
        ax.imshow(im[crop]);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(RESULTS/'gradient_seam_detail.png',dpi=180,bbox_inches='tight');plt.close(fig)


def plot_panorama(views,pano_sift,pano_orb):
    fig=plt.figure(figsize=(14,8))
    grid=fig.add_gridspec(2,6,height_ratios=[1,1.15],hspace=.22,wspace=.08)
    for i,view in enumerate(views):
        ax=fig.add_subplot(grid[0,2*i:2*i+2]);ax.imshow(view);ax.set_title(f'View {i+1}');ax.axis('off')
    ax=fig.add_subplot(grid[1,0:3]);ax.imshow(pano_sift);ax.set_title('Harris-NCC panorama');ax.axis('off')
    ax=fig.add_subplot(grid[1,3:6]);ax.imshow(pano_orb);ax.set_title('ORB-style panorama');ax.axis('off')
    fig.suptitle('Feature-based stitching on a generated three-view scene',fontsize=14)
    fig.savefig(RESULTS/'panorama_results.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    # Part 1: direct baseline and discrete Poisson solve.
    source,mask,dest,offset=make_composite_pair()
    direct=direct_clone(source,mask,dest,offset)
    t0=time.perf_counter();poisson=poisson_clone(source,mask,dest,offset);poisson_time=time.perf_counter()-t0
    save_rgb(source,ASSETS/'gradient_source_synthetic.png');save_rgb(dest,ASSETS/'gradient_background_synthetic.png')
    Image.fromarray(mask).save(ASSETS/'gradient_manual_mask.png')
    save_rgb(direct,RESULTS/'gradient_direct.png');save_rgb(poisson,RESULTS/'gradient_poisson.png')
    metrics={'gradient_domain':{'mask_pixels':int((mask>0).sum()),'direct_boundary_edge_mean_l1':seam_edge_mean(direct,mask,offset),
      'poisson_boundary_edge_mean_l1':seam_edge_mean(poisson,mask,offset),'poisson_runtime_s':poisson_time}}
    plot_composites(dest,direct,poisson,mask,offset)
    # Part 2: one common pipeline for two descriptor families.
    scene=make_panorama_scene();views,Hs=generate_panorama_views(scene)
    sift,ps,_=run_feature_method(views,Hs,'Harris-NCC')
    orb,po,_=run_feature_method(views,Hs,'ORB-style')
    metrics['stitching']={'synthetic_view_count':len(views),'view_shape':list(views[0].shape[:2]),
       'ground_truth_transform_note':'Known projective transforms used only to calculate homography corner RMSE.',
       'methods':[sift,orb]}
    plot_panorama(views,ps,po)
    (RESULTS/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
