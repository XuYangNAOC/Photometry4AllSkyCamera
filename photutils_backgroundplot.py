import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord,get_body
from astropy.time import Time
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from astropy.stats import SigmaClip,sigma_clipped_stats
from photutils.aperture import CircularAperture, aperture_photometry,ApertureStats,CircularAnnulus
from astropy.io import fits
from scipy.stats import norm
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, ModeEstimatorBackground, SExtractorBackground
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceFinder, SourceCatalog
from astropy.convolution import convolve
from datetime import datetime
from matplotlib.patches import Circle

import time
starttime=time.time()
sigclip = SigmaClip(sigma=3, maxiters=5)

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


def ds9output(x,y,name,shape='circle',size=6,fill=0,color='green'):
    f=open(name+'.reg','w+')
    print('# Region file format: DS9 version 4.1',file=f)
    print('global color='+color+' dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',file=f)
    print('physical',file=f)
    if(shape=='circle'):
        for i in range(len(x)):
            if(np.isnan(x[i]))or(np.isnan(y[i])):
                continue
            print('circle(%.02f,%.02f,%i) # fill=%i'%(x[i],y[i],size,fill),file=f)
    elif(shape=='box'):
        for i in range(len(x)):
            print('box(%.02f,%.02f,%i,%i,0) # fill=%i'%(x[i],y[i],size,size,fill),file=f)
    else:print('unknown shape:'+shape+'. only circle and box supported')
    f.close()


# photutils
def CircleStat(x0,y0,r_in,r_out,data,target='test',mode='draw',bkgm1=0,):
    if(mode=='subpixel'):
        subpixel_size = 5
        subpixel_matrix = np.ones((subpixel_size, subpixel_size), dtype=data.dtype)
        split_image = np.kron(data, subpixel_matrix)
        data = split_image
        x0 = x0 * subpixel_size + (subpixel_size-1)/2
        y0 = y0 * subpixel_size + (subpixel_size-1)/2
        r_in = r_in * subpixel_size
        r_out = r_out * subpixel_size
    mask = np.zeros_like(data, dtype=bool)
    y, x = np.indices(data.shape)
    mask_circle_inner = (x - x0)**2 + (y - y0)**2 >= r_in**2
    mask_circle_outer = (x - x0)**2 + (y - y0)**2 <= r_out**2
    mask_annulus = mask_circle_outer & mask_circle_inner  # 圆环形区域
    mask[mask_annulus] = True
    annulus_region = data[mask]
    c_annu=sigclip(annulus_region)
    #print(annulus_region,c_annu)
    params=norm.fit(c_annu)
    if(mode=='draw'):
        plt.figure(1)
        left=min(annulus_region)
        right=max(annulus_region)
        len_bins=int((right-left)/5)
        n1,bins1,patches1=plt.hist(annulus_region,density=True,bins=len_bins,label='mode=%i std=%.02f'%(params[0],params[1]))
        bins_w1= bins1[1] - bins1[0]
        bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
        popt1, pcov1 = curve_fit(gaussian, bin_centers1, n1, 
                                 p0=[1, np.mean(annulus_region), np.std(annulus_region)])
        x1 = np.linspace(left, right, 500)
        plt.plot(x1, gaussian(x1, *popt1), 'r-', linewidth=2, 
                 label=f'Gaussian fit: μ={popt1[1]:.2f}, σ={popt1[2]:.2f}')
        plt.xlabel('background')
        plt.ylabel('count')
        plt.legend()

        #plt.xlim(popt1[1]-5*popt1[2]*1.5,popt1[1]+5*popt1[2]*1.5)

        plt.title(target+' %i %i'%(x0,y0))
        plt.savefig('plot/'+target[0:22]+'/%.01f_%.01f_%02i.png'%(x0,y0,r_out))
        f=open(target[0:22]+'%02i.txt'%r_out,'a+')
        print("%.01f %.01f %.01f %0.2f"%(x0,y0,popt1[1],bkgm1),file=f)
        f.close()
        plt.close()

    return annulus_region


from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
def crossmatch(p1,p2,max_dist=1.0):
    dist_matrix=cdist(p1,p2,'euclidean')
    mask = dist_matrix <= max_dist
    #print(len(dist_matrix[mask]))
    #print(mask)
    # 无有效匹配时的处理
    if not np.any(mask):
        return [], []  # 返回空匹配
    
    # 仅保留有效距离并求解
    valid_dist = np.where(mask, dist_matrix, np.inf)
    row_ind, col_ind = linear_sum_assignment(valid_dist)
    
    # 二次过滤确保结果有效
    valid_pairs = [(i,j) for i,j in zip(row_ind, col_ind) 
                  if not np.isinf(valid_dist[i,j])]
    return zip(*valid_pairs) if valid_pairs else ([], [])

def circle_mask(x,y,x_c,y_c,r,mode='out'):
    Y,X = np.ogrid[:x,:y]
    dist = np.sqrt((X-x_c)**2+(Y-y_c)**2)
    if(mode=='out'):return dist>r
    else:return dist<=r

def taylor(xdata,k1,k2,k3,k4):
    return k1*xdata+k2*(xdata**3)+k3*(xdata**5)+k4*(xdata**7)

def xytoradec(x,y,x0,y0,E,a0,e,k1,k2,k3,k4,time0,loc):
    cz=taylor(np.sqrt((x-x0)**2+(y-y0)**2),k1,k2,k3,k4)
    ca=np.arctan2(x-x0,y-y0)+np.pi
    u0=cz
    X=a0-E
    b=X+ca
    Cze=np.arccos(np.cos(u0)*np.cos(e)-np.cos(b)*np.sin(u0)*np.sin(e))

    amE = np.arctan2(np.sin(b)*np.sin(u0),(np.cos(b)*np.sin(u0)*np.cos(e)+np.cos(u0)*np.sin(e)))
    Caz=[]
    for lines in amE:
        Caz0=[x+E if x+E >0 else x+E+np.pi*2 for x in lines]
        Caz.append(Caz0)
    Caz=np.array(Caz)

    Calt=np.pi/2-Cze
    time = Time(time0)
    altaz_from_xy=AltAz(obstime=time,location=loc,az=Caz*u.rad,alt=Calt*u.rad)
    skycord_from_xy=SkyCoord(altaz_from_xy)
    #radec_from_xy=skycord_from_xy.transform_to('icrs')
    #ra=radec_from_xy.ra.degree
    #dec=radec_from_xy.dec.degree
    galactic=skycord_from_xy.galactic
    l_cord=galactic.l.degree
    b_cord=galactic.b.degree
    return skycord_from_xy,l_cord,b_cord

def galac_mask(x,y,x_c,y_c,r,path,mode='out'):
    Y,X = np.ogrid[:x,:y]
    
    #dist = np.sqrt((X-x_c)**2+(Y-y_c)**2)
    #print(dist)
    path2=path[0:26]
    E,a0,e0,k1,k2,k3,k4=np.loadtxt('d:/seafile/私人资料库/photutils/astropara/'+path2+'_parameters.txt')
    x0,y0=np.loadtxt('x0y0.txt')
    time0 =path2[6:10]+'-'+path2[10:12]+'-'+path2[12:14]+' '+path2[14:16]+':'+path2[16:18]+':'+path2[18:20]
    #print(time0)
    #exit()
    time1 = Time(time0)
    domea = EarthLocation(lat=-80.41694*u.deg, lon=77.11611*u.deg, height=4093*u.m)
    skycord,l,b=xytoradec(X,Y,x0,y0,E,a0,e0,k1,k2,k3,k4,time1,domea)
    #print(skycord)
    # distance to the moon
    #SEP_C=[]
    if(mode=='moon'):
        moon=get_body('moon',time1,location=domea)
        sep_c=skycord.separation(moon).degree
        #max_sep=15*60*u.arcmin
        #print(sep_c)
        #for i in distc:
            #print(i)
            #idx,d2d,d3d=i.match_to_catalog_3d(moon)
            #max_sep=15*60*u.arcmin
            #sep_c=d2d<=max_sep
            #SEP_C.append(sep_c)
        #SEP_C=np.array(SEP_C)
    #print(b.shape,sep_c.shape)
    if(mode=='moon'):
        return (abs(b)<=r)|(sep_c<=35)
    else:
        return abs(b)<=r

def tdback(data,s1=25,s2=1,mask=None,threshold=None):
    #bkgest=MedianBackground()
    bkgest=SExtractorBackground(sigma_clip=sigclip)
    bkg=Background2D(data,(s1,s1),filter_size=(s2,s2),sigma_clip=sigclip,bkg_estimator=bkgest,coverage_mask=mask,filter_threshold=threshold)
    return bkg

def starfind(path,x_c=1338,y_c=910,r_c=700,areamap=0,zenithmap=0,mode='in',iso=1600,buff=0):
    path0=path.split('/')[-1]
    path2=path0[:-11]+'Blue.fits'
    
    #print(path0,path,path2)
    #exit()
    Bb,Db,Bg,Dg,Ca,Cb=np.loadtxt('correction_parameters.txt',unpack=True)
    t_a,t_b,t_c=np.loadtxt('param_timecorr.txt',unpack=True)
    tb_a,tb_b,tb_c=np.loadtxt('param_timecorr_B.txt',unpack=True)
    date0=datetime.strptime('20170329190002','%Y%m%d%H%M%S').timestamp()
    date1=datetime.strptime(path2[6:20],'%Y%m%d%H%M%S').timestamp()
    date_corr=date1-date0
    p=np.poly1d([t_a,t_b,t_c])
    pb=np.poly1d([tb_a,tb_b,tb_c])
    Dg=Dg+p(date_corr)
    Db=Db+pb(date_corr)
    
    with fits.open(path) as hdul:
        data_b=hdul[0].data
   
    with fits.open(path) as hdul:
        data=hdul[0].data
        #mask=circle_mask(data.shape[0],data.shape[1],x_c,y_c,r_c)
        #mask[841:970,1775:2040]=True

        mask1=circle_mask(data.shape[0],data.shape[1],x_c,y_c,650)
        mask2=galac_mask(data.shape[0],data.shape[1],x_c,y_c,10,path=path0,mode=mode)
        bkg=tdback(data,s1=25,s2=1,threshold=None)
        
        bkg2=tdback(data_b,s1=25,s2=1,threshold=None)

        iso=1600
        zp_flux=63095.7315
        iso_corr=1600.0/iso
        area_m=163164.791
        
        back_arcsec=bkg.background*iso_corr/areamap
        backb_arcsec=bkg2.background*iso_corr/areamap
        back_arcsec0=bkg.background*iso_corr/area_m
        
        magback=-2.5*np.log10(back_arcsec/zp_flux*iso_corr)
        magback=magback+Bg*np.square(zenithmap)+Dg
        magbback=-2.5*np.log10(backb_arcsec/zp_flux*iso_corr)
        magbback=magbback+Bb*np.square(zenithmap)+Db
        magback0=-2.5*np.log10(back_arcsec0/zp_flux*iso_corr)
        
        mag_plot=np.ma.masked_where(mask1,magback)
        magb_plot=np.ma.masked_where(mask1,magbback)
        mag_plot0=np.ma.masked_where(mask1,magback0)
        #back_plot=np.ma.masked_where(mask1|mask2,bkg.background*iso_corr)
        back_plot=np.ma.masked_where(mask1,bkg.background*iso_corr)
        back_plot=np.ma.masked_where(mask1,bkg2.background*iso_corr)

        magv_plot=(1-Ca)*magb_plot+Ca*mag_plot-Cb

        """
        plt.rcParams['font.size'] = 14
        plt.figure(1+buff)
        #plt.title(path0+' s1=25,s2=1')
        plt.title(path0.split('.')[0]+' Background (V)')#,fontsize=12)
        # 转到和原图相同
        plt.imshow(np.fliplr(np.flipud(magv_plot.T)),origin='lower', cmap='RdYlBu',interpolation='nearest')
        #plt.imshow(np.flipud(magv_plot.T),origin='lower', cmap='RdYlBu',interpolation='nearest')
        #plt.imshow(np.flipud(magv_plot.T),vmin=18.5,vmax=21.5,origin='lower', cmap='RdYlBu',interpolation='nearest')
        circle = Circle([629,1330], 134, facecolor='none', alpha=1, edgecolor='black', linewidth=3)
        plt.gca().add_patch(circle)
        cb1=plt.colorbar()
        cb1.set_label(r'mag arcsec$^{-2}$')
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.xlim(680,2010)
        #plt.ylim(240,1580)
        plt.ylim(640,1970)
        plt.xlim(180,1520)
        plt.text(564,1252,'SMC',fontsize=13,color='red')
        plt.text(618,1070,'LMC',fontsize=13,color='red')
        plt.text(1132,979,'Milky Way',fontsize=13,color='red',rotation=270)
        

        plt.figure(2+buff)
        plt.title(path0+' Background (G)')
        plt.imshow(np.flipud(mag_plot0.T),origin='lower', cmap='RdYlBu',interpolation='nearest')
        plt.imshow(np.flipud(mag_plot0.T),vmin=18.5,vmax=21.5,origin='lower', cmap='RdYlBu',interpolation='nearest')
        cb2=plt.colorbar()
        cb2.set_label(r'mag arcsec$^{-2}$')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ylim(640,1970)
        plt.xlim(240,1580)
        
        #plt.figure(2+buff)
        #plt.title(path[5:31]+' s1=25,s2=1')
        #plt.imshow(back_plot,origin='lower', cmap='RdYlBu_r',interpolation='nearest')
        #plt.colorbar()
        """
        
        """
        # 对比两种取背景的方法（s2是否取filter）
        back_arcsec2=bkg2.background*iso_corr/((7*60)**2)
        magback2=-2.5*np.log10(back_arcsec2/zp_flux)
        mag_plot2=np.ma.masked_where(mask1,magback2)
        #back_plot2=np.ma.masked_where(mask1|mask2,bkg2.background*iso_corr)
        back_plot2=np.ma.masked_where(mask1,bkg2.background*iso_corr)

        plt.figure(3+buff)
        plt.title(path[5:31]+' s1=25,s2=5')
        plt.imshow(mag_plot2,origin='lower', cmap='RdYlBu',interpolation='nearest')
        plt.colorbar()

        plt.figure(4+buff)
        plt.title(path[5:31]+' s1=25,s2=5')
        plt.imshow(back_plot2,origin='lower', cmap='RdYlBu_r',interpolation='nearest')
        plt.colorbar()

        plt.figure(5+buff)
        back_plot3=np.ma.masked_where(mask1,bkg.background/bkg2.background)
        plt.title(path[5:31]+' s1=25, s2=1 / s2=5')
        plt.imshow(back_plot3,origin='lower', cmap='RdYlBu_r',interpolation='nearest')
        plt.colorbar()
        
        #print(back_plot)

        plt.figure(6+buff)
        hist,edges=np.histogram(back_plot3[~np.isnan(back_plot3)],bins=140,range=(-1,6))
        plt.bar(edges[:-1],hist,width=np.diff(edges)[0], align='edge')
        #plt.hist(back_plot[back_plot>0],range=(0,1400),bins=200)
        #plt.show()
        #plt.close()
        """
        
    #with fits.open(path2+'Red.fits') as hdu2:
    #    data2=hdu2[0].data
    #    bkg2=tdback(data2,s1=25,s2=1,threshold=None)

    #    back_arcsec2=bkg2.background*iso_corr/((7*60)**2)
    #    magback2=-2.5*np.log10(back_arcsec2/zp_flux*iso_corr)
    #    mag_plot2=np.ma.masked_where(mask1,magback2)
    #    back_plot2=np.ma.masked_where(mask1,bkg2.background*iso_corr)
        
    #with fits.open(path2+'Blue.fits') as hdu3:
    #    data3=hdu3[0].data
    #    bkg3=tdback(data3,s1=25,s2=1,threshold=None)

    #    back_arcsec3=bkg3.background*iso_corr/((7*60)**2)
    #    magback3=-2.5*np.log10(back_arcsec3/zp_flux*iso_corr)
    #    mag_plot3=np.ma.masked_where(mask1,magback3)
    #    back_plot3=np.ma.masked_where(mask1,bkg3.background)

    #plt.figure(3+buff)
    #plt.title(path[5:31]+' s1=25,s2=1')
    #plt.imshow(mag_plot2,origin='lower', cmap='RdYlBu',interpolation='nearest')
    #cb1=plt.colorbar()
        
    #plt.figure(4+buff)
    #plt.title(path[5:31]+' s1=25,s2=1')
    #plt.imshow(mag_plot3,origin='lower', cmap='RdYlBu',interpolation='nearest')
    #plt.colorbar()
        
    return 0
    

r_in=6
r_out=12

areamap=np.loadtxt('area_map.txt')
zenithmap=np.loadtxt('zenith_map.txt')

for k in range(0,1):
    r_out=k
    # clear
    target1='fits/KLCAM_20170505200002_1.cr2Green1.fits'
    target8='fits/KLCAM_20170520170000_1.cr2Green1.fits'
    # cloudy
    target2='fits/KLCAM_20170807223050_1.cr2Green1.fits'
    # dawn
    target3='fits/KLCAM_20170329190002_1.cr2Green1.fits'
    # moon1
    target4='fits/KLCAM_20170409200001_1.cr2Green1.fits'
    #target4='fits/KLCAM_20170409200001_1.cr2Red.fits'
    #target4='fits/KLCAM_20170409200001_1.cr2Blue.fits'
    # moon2
    target5='fits/KLCAM_20170416210003_1.cr2Green1.fits'
    # aurora1
    target6='fits/KLCAM_20170521230001_1.cr2Green1.fits'
    # aurora2
    target7='fits/KLCAM_20170528180003_1.cr2Green1.fits'

   
    catalog4=starfind('../'+target8,areamap=areamap,zenithmap=zenithmap)
    #catalog4=starfind(target7,mode='moon',buff=6)

    #plt.show()
    endtime=time.time()
    print(endtime-starttime)
    #x=catalog1['xcentroid']+1
    #y=catalog1['ycentroid']+1
    #mag=catalog1['mag_p1']
    #ds9output(x,y,name='reg/'+target1.split('.')[0][5:]+'.photutils.s1.25',color='red')
    #np.savetxt(np.array([x,y,mag]).T,fmt='.02f')

    """
    catalog2=starfind(target2)
    x=catalog2['xcentroid']+1
    y=catalog2['ycentroid']+1
    #ds9output(x,y,name='reg/'+target2.split('.')[0][5:]+'.photutils.s1.25',color='red')

    
    catalog3=starfind(target3)
    x=catalog3['xcentroid']+1
    y=catalog3['ycentroid']+1
    ds9output(x,y,name='reg/'+target3.split('.')[0][5:]+'.photutils.s1.25',color='red')
    
    catalog4=starfind(target4)
    x=catalog4['xcentroid']+1
    y=catalog4['ycentroid']+1
    ds9output(x,y,name='reg/'+target4.split('.')[0][5:]+'.photutils.s1.25',color='red')

    catalog5=starfind(target5)
    x=catalog5['xcentroid']+1
    y=catalog5['ycentroid']+1
    ds9output(x,y,name='reg/'+target5.split('.')[0][5:]+'.photutils.s1.25',color='red')
    
    catalog6=starfind(target6)
    x=catalog6['xcentroid']+1
    y=catalog6['ycentroid']+1
    ds9output(x,y,name='reg/'+target6.split('.')[0][5:]+'.photutils.nomask',color='red')
    
    catalog7=starfind(target7)
    x=catalog7['xcentroid']+1
    y=catalog7['ycentroid']+1
    ds9output(x,y,name='reg/'+target7.split('.')[0][5:]+'.photutils.s1.25',color='red')
    """
