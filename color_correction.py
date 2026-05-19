import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from astropy.stats import SigmaClip
from photutils.aperture import CircularAperture, aperture_photometry,ApertureStats,CircularAnnulus
from astropy.io import fits
from scipy.stats import norm

sigclip = SigmaClip(sigma=3.0, maxiters=10)

def t3fit(x,a,b,c,d,e):
#def t3fit(x,b,d):
    #print(x)
    return b*x**2+d+c*x+a*x**3+e*x**4
    #return b*x**2+d

#def t3fit(x,a,b,c,d,e):
def t2fit(x,b,d):
    #print(x)
    #return b*x**2+d+c*x+a*x**3+e*x**4
    return b*x**2+d

def linear(x,a,b):
    return a*x+b

def crossmatch(ra1,dec1,ra2,dec2,maxstep):
    wcs1=SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree)
    wcs2=SkyCoord(ra=ra2*u.degree,dec=dec2*u.degree)
    idx,d2d,d3d=wcs1.match_to_catalog_3d(wcs2)
    max_sep=maxstep*u.arcmin
    sep_c=d2d<max_sep
    return sep_c,idx[sep_c]

def ds9output(x,y,name,color='green',size=10):
    f=open(name+'.reg','w+')
    print('# Region file format: DS9 version 4.1',file=f)
    print('global color='+color+' dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',file=f)
    print('physical',file=f)
    for i in range(len(x)):
        print('circle(%.02f,%.02f,%i)'%(x[i],y[i],size),file=f)
    f.close()
    
def single(path1='cali_back16/KLCAM_20170620000016_1.cr2_aper.cat',
           #path2='cali_back16/KLCAM_20170620003016_1.cr2_aper.cat',
           Ca=0,
           Cb=0,
           mode='test'):
    
    Rra,Rdec,Rmag,RBmag=np.loadtxt('standard_catalogue/tycho_v6_novary.v2.txt',skiprows=1,dtype=float,unpack=True)

    data1=np.loadtxt(path1,unpack=True)
    #data2=np.loadtxt(path2,unpack=True)

    x1,y1=data1[0:2]
    ra1,dec1=data1[2:4]
    mag21,mag41,mag61,mag81,mag101=data1[4:9]
    mag_iso1=data1[10]
    mag_auto1=data1[9]
    fwhm1=data1[11]
    flux21,flux41,flux61,flux81,flux101=data1[12:17]
    azimuth1=data1[17]
    zenith1=data1[18]
    #back1=data1[19]
    back1=data1[20]

    index=np.isnan(mag61)

    ra1=ra1[~index]
    dec1=dec1[~index]
    x1=x1[~index]
    y1=y1[~index]
    mag21=mag21[~index]
    mag41=mag41[~index]
    mag61=mag61[~index]
    mag81=mag81[~index]
    mag101=mag101[~index]
    fwhm1=fwhm1[~index]
    flux21=flux21[~index]
    flux41=flux41[~index]
    flux61=flux61[~index]
    flux81=flux81[~index]
    flux101=flux101[~index]
    mag_iso1=mag_iso1[~index]
    mag_auto1=mag_auto1[~index]
    zenith1=zenith1[~index]
    azimuth1=azimuth1[~index]
    back1=back1[~index]
    #x2,y2=data2[0:2]
    #ra2,dec2=data2[2:4]
    #mag22,mag42,mag62,mag82,mag102=data2[4:9]
    #mag_iso2=data2[10]
    #mag_auto2=data2[9]
    #fwhm2=data2[11]
    #flux22,flux42,flux62,flux82,flux102=data2[12:17]
    #azimuth2=data2[17]
    #zenith2=data2[18]

    # compare

    wcs1=SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree)
    #wcs2=SkyCoord(ra=ra2*u.degree,dec=dec2*u.degree)
    wcs3=SkyCoord(ra=Rra*u.degree,dec=Rdec*u.degree)

    idx,d2d,d3d=wcs1.match_to_catalog_3d(wcs3)
    max_sep=21*u.arcmin
    sep_c=d2d<max_sep

    if(path1.split('.')[1]=='b'):
        mag_tycho=RBmag[idx[sep_c]]
    else:
        mag_tycho=Rmag[idx[sep_c]]
        mag_B=RBmag[idx[sep_c]]
    
    Rra_1=Rra[idx[sep_c]]
    Rdec_1=Rdec[idx[sep_c]]

    ra_1=ra1[sep_c]
    dec_1=dec1[sep_c]
    x_1=x1[sep_c]
    y_1=y1[sep_c]
    mag_21=mag21[sep_c]
    mag_41=mag41[sep_c]
    mag_61=mag61[sep_c]
    mag_81=mag81[sep_c]
    mag_101=mag101[sep_c]
    fwhm_1=fwhm1[sep_c]
    flux_101=flux101[sep_c]
    mag_iso_1=mag_iso1[sep_c]
    mag_auto_1=mag_auto1[sep_c]
    zenith_1=zenith1[sep_c]
    azimuth_1=azimuth1[sep_c]
    back_1=back1[sep_c]
    #print(len(mag_tycho),len(mag_101))
    mag_101=mag_61
    mago_101=mag_61
    if(mode=='color'):
        mag_101=(1-Ca)*mag_B+Ca*mag_61-Cb
        #print(Ca,Cb)
    #"""
    #if not, return error. should be written in a function later
    if(mode=='65')or(mode=='color'):
        #index1=set(list(np.where(zenith_1>65)[0]))
        #index2=set(list(np.where(mag_101>5)[0]))
        #index=np.array(list(index1|index2))
        index=np.where(zenith_1>65)[0]
        mag_101=np.delete(mag_101,index)
        mago_101=np.delete(mago_101,index)
        mag_tycho=np.delete(mag_tycho,index)
        azimuth_1=np.delete(azimuth_1,index)
        zenith_1=np.delete(zenith_1,index)
        x_1=np.delete(x_1,index)
        y_1=np.delete(y_1,index)
        ra_1=np.delete(ra_1,index)
        dec_1=np.delete(dec_1,index)
        Rra_1=np.delete(Rra_1,index)
        Rdec_1=np.delete(Rdec_1,index)
        back_1=np.delete(back_1,index)
    #print(index)
    #"""
    
    magoc=mag_tycho-mag_101
    #print(magoc)
    #popt,pcov=curve_fit(t3fit,zenith_1,magoc)
    #a,b,c,d,e=popt
    if(len(magoc)<=1):
        a=0
        b=1
        c=0
        e=0
        if(len(magoc)==0):
            d=0
            num=0
            std=0
        else:
            d=magoc[0]
            num=len(magoc)
            std=np.std(magoc)
        return num,mag_101,mago_101,mag_tycho,back_1,azimuth_1,zenith_1,x_1,y_1,ra_1,dec_1,Rra_1,Rdec_1,magoc,a,b,c,d,e,std
    popt,pcov=curve_fit(t2fit,zenith_1,magoc)
    b,d=popt                                    # 1

    #"""
    #mag_101new=mag_101+t3fit(zenith_1,a,b,c,d,e)
    mag_101new=mag_101+t2fit(zenith_1,b,d)      # 2
    magoc_new=mag_tycho-mag_101new
    magoc_clip=sigclip(magoc_new)
    #print(mag_101clip)
    index=np.where(magoc_new==magoc_clip)[0]
    if(mode=='color'):
        index_inv=np.where(magoc_new!=magoc_clip)[0]
        wcs_c=SkyCoord(ra=ra_1*u.degree,dec=dec_1*u.degree)
        galactic=wcs_c.galactic
        Rb=galactic.b.to_value('degree')
        index1=set(list(index_inv))
        index2=set(list(np.where(abs(Rb)>10)[0]))
        index_inv2=np.array(list(index1&index2))
        #wcs_t=SkyCoord(ra=Rra_1[index_inv]*u.degree,dec=Rdec_1[index_inv]*u.degree)
        #l,b
        #for i in range(len(x_1[index_inv2])):
        #    print('%.01f %.01f %.02f %.02f %.04f %.04f'%(x_1[index_inv2][i],y_1[index_inv2][i],mag_101[index_inv2][i],mag_101[index_inv2][i]-mag_tycho[index_inv2][i],Rra_1[index_inv2][i],Rdec_1[index_inv2][i]))
        #ds9output(x_1[index_inv],y_1[index_inv],name=tar+'_fitting',color='green',size=10)
        #ds9output(x_1[index_inv2],y_1[index_inv2],name=tar+'_b10',color='yellow',size=8)
    num=len(index)
    #if(num<1000):
    #    return 0
    mag_tychofit=mag_tycho[index]
    mag_101fit=mag_101[index]

    magoc=mag_tychofit-mag_101fit
    zenith_fit=zenith_1[index]
    #azimuth_1=azimuth_1[index]
    #zenith_1=zenith_1[index]
    #ra_1=ra_1[index]
    #dec_1=dec_1[index]
    #x_1=x_1[index]
    #y_1=y_1[index]
    back_1=back_1[index]
    indexx=np.where(zenith_fit<=65)[0]
    back_p=back_1[indexx]
    a=0
    c=0
    e=0
    #popt,pcov=curve_fit(t3fit,zenith_fit,magoc)
    #a,b,c,d,e=popt
    popt,pcov=curve_fit(t2fit,zenith_fit,magoc)
    b,d=popt                                    # 3
    std=np.std(magoc-t2fit(zenith_fit,b,d))
    #"""
    
    return num,mag_101,mago_101,mag_tycho,back_p,azimuth_1,zenith_1,x_1,y_1,ra_1,dec_1,Rra_1,Rdec_1,magoc,a,b,c,d,e,std

def colorcoef(path,mode='test'):
    tar1,_1,color1,_2=path.split('.')
    tar=tar1+'.'+_1
    rr_b=single(path1='fitback/'+tar+'.b.cat',mode='65')
    rr_g=single(path1='fitback/'+tar+'.g.cat',mode='65')
    rr_r=single(path1='fitback/'+tar+'.r.cat',mode='65')
    num,mag_1,mago_1,magb_tycho,backb,azimuth_1,zenith_1,x_1,y_1,ra_1,dec_1,Rra_1,Rdec_1,magocb,a1,b1,c1,d1,e1,std1=rr_b
    num,mag_2,mago_2,magg_tycho,backg,azimuth_2,zenith_2,x_2,y_2,ra_2,dec_2,Rra_2,Rdec_2,magocg,a2,b2,c2,d2,e2,std2=rr_g
    num3,mag_3,mago_3,magr_tycho,backr,azimuth_3,zenith_3,x_3,y_3,ra_3,dec_3,Rra_3,Rdec_3,magocr,a3,b3,c3,d3,e3,std3=rr_r
    print(b1,d1)
    wcs1=SkyCoord(ra=ra_1*u.degree,dec=dec_1*u.degree)
    wcs2=SkyCoord(ra=ra_2*u.degree,dec=dec_2*u.degree)
    if(len(ra_1)==0)or(len(ra_2)==0):
        plt.figure(1)
        plt.title('no star detected')
        plt.savefig('figure_fit/figure_colorterm/'+tar+'_%.02f_%.02f_b-v.png'%(0,0))
        plt.close()
        return 0
    idx,d2d,d3d=wcs1.match_to_catalog_3d(wcs2)
    max_sep=21*u.arcmin
    sep_c=d2d<max_sep
    
    mag1=mag_1[sep_c]+t2fit(zenith_1[sep_c],b1,d1)
    Rmag1=magb_tycho[sep_c]
    mag2=mag_2[idx[sep_c]]+t2fit(zenith_2[idx[sep_c]],b2,d2)
    Rmag2=magg_tycho[idx[sep_c]]

    mag_v=mag_2[idx[sep_c]]
    
    azimuth_2=azimuth_2[idx[sep_c]]
    zenith_2=zenith_2[idx[sep_c]]

    BG=mag1-mag2
    BV=mag2-Rmag2
    if(len(BG)<=1):
        return 1,0,mag_v,mag2,mag2,Rmag2,azimuth_2,zenith_2,backr,backg,backb,b1,d1,b2,d2
    #print(len(BG),len(BV))
    popt,pcov=curve_fit(linear,BG,BV)
    Ca,Cb=popt

    BG_c=linear(BG,Ca,Cb)
    oc=sigclip(BG_c-BV)
    index=np.where(BG_c-BV==oc)[0]
    BV_cc=BV[index]
    BG_cc=BG[index]
    
    # image check
    #index_inv1=set(list(np.where(BG_c-BV!=oc)[0]))
    #index_inv2=set(list(np.where(BV<-0.7)[0]))
    #index_inv=np.array(list(index_inv1&index_inv2))
    #index_inv=np.where(BG_c-BV!=oc)[0]
    #for i in range(len(x_2[index_inv])):
    #    print('%.01f %0.1f %.02f %.02f %.02f %.02f %.04f %.04f'%(x_2[index_inv][i],y_2[index_inv][i],mag1[index_inv][i],Rmag1[index_inv][i],mag2[index_inv][i],Rmag2[index_inv][i],Rra_2[index_inv][i],Rdec_2[index_inv][i]))
    #ds9output(x_2[index_inv],y_2[index_inv],name=tar+'_color',color='red',size=12)
    
    popt,pcov=curve_fit(linear,BG_cc,BV_cc)
    Ca,Cb=popt
    perr = np.sqrt(np.diag(pcov))
    eCa,eCb=perr
    print(popt)
    print(pcov)
    print(perr)

    px=np.arange(min(BG_cc)-0.01,max(BG_cc)+0.01,0.01)

    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 14
    # img check
    plt.figure(1)
    plt.plot(BG,BV,'b.',label='3sigma outliers')
    plt.plot(BG_cc,BV_cc,'r.')
    plt.plot(px,linear(px,Ca,Cb),'k-',label=r'$(G_{Canon}-V_{Tycho}) = (%.02f\pm%.02f)\times(B_{Canon}-G_{Canon}) + (%.02f\pm%.03f)$'%(Ca,eCa,Cb,eCb))
    plt.xlabel(r'$B_{Canon} - G_{Canon}$ (mag)')
    plt.ylabel(r'$G_{Canon} - V_{Tycho}$ (mag)')
    plt.title(path[:-14])
    plt.legend()
    plt.grid()
    #plt.savefig('figure_fit/figure_colorterm/'+tar+'_%.02f_%.02f_b-v.png'%(Ca,Cb))
    plt.show()
    plt.close()

    mag_v2=mag2-Ca*(mag1-mag2)-Cb
    print(path)
    print('blue:     b=',b1,'d=',d1)
    print('green:    b=',b2,'d=',d2)
    print('color:   Ca=',Ca,'Cb=',Cb)
    print('mag_new=mag_old+b*zenith^2+d')
    #print('mag_v=(1-Ca)*mag_b+Ca*mag_g-Cb')
    print('mag_v=mag_g-Ca*(mag_b-mag_g)-Cb')

    """
    magoc=Rmag2-mag_v
    popt,pcov=curve_fit(t2fit,zenith_2,magoc)
    b,d=popt                           
    mag_101new=mag_v+t2fit(zenith_2,b,d)      # 2
    magoc_new=Rmag2-mag_101new
    magoc_clip=sigclip(magoc_new)
    index=np.where(magoc_new==magoc_clip)[0]
    
    mag_tychofit=Rmag2[index]
    mag_101fit=mag_v[index]

    magoc=mag_tychofit-mag_101fit
    zenith_fit=zenith_2[index]
    
    #back_1=back_1[index]
    #indexx=np.where(zenith_fit<=65)[0]
    #back_p=back_1[indexx]
    a=0
    c=0
    e=0
    #popt,pcov=curve_fit(t3fit,zenith_fit,magoc)
    #a,b,c,d,e=popt
    popt,pcov=curve_fit(t2fit,zenith_fit,magoc)
    b,d=popt

    mag_v2=mag_v+t2fit(zenith_2,b,d)
    """
    
    return Ca,Cb,mag_v,mag_v2,mag2,Rmag2,azimuth_2,zenith_2,backr,backg,backb,b1,d1,b2,d2
    

#"""
# main loop

#f=open('statslist_fitback.txt','w+')
lists=np.loadtxt('colourlist.txt',unpack=True,dtype='str')
lists=['KLCAM_20170505200002_1.b.cat']
i=0
for path in lists:
    color=path.split('.')[1]
    tar=path.split('.')[0]+'.backfit'
    path=tar+'.'+color+'.cat'
    if(color!='b'):continue
    rc=colorcoef(path)
    if(rc==0):
        plt.figure(2)
        plt.title('no star detected')
        plt.savefig('figure_fit/figure_fit_colorcorr/'+path+'.png')
        plt.close()
        #print(path,0,0,0,0,0,0,0,0,0,0,file=f)
        continue
    Ca,Cb,mag_v,mag_v2,mag2,Rmag2,azimuth_2,zenith_2,backr,backg,backb,bb,db,b,d=rc
    rr=single(path1='fitback/'+tar+'.g.cat',Ca=Ca,Cb=Cb,mode='color')
    num,mag_1,mago_1,mag_tycho,back,azimuth_1,zenith_1,x_1,y_1,ra_1,dec_1,Rra_1,Rdec_1,magoc,a1,b1,c1,d1,e1,std1=rr
    
    # single target check
    #wcs1=SkyCoord(ra=Rra_1*u.degree,dec=Rdec_1*u.degree)
    #wcs2=SkyCoord(ra=359.3874*u.degree,dec=-82.1697*u.degree)
    #print(wcs1,wcs2)
    #idx,d2d,d3d=wcs1.match_to_catalog_3d(wcs2)
    #max_sep=21*u.arcmin
    #sep_c=d2d<max_sep
    #print(x_1[sep_c],y_1[sep_c],mag_1[sep_c],mag_tycho[sep_c])
    
    magoc=mag_tycho-mag_1
    back_p=back
    num_p=len(back_p)
    #print(path,num_p,np.median(backr),np.median(backg),np.median(backb),b,d,bb,db,Ca,Cb,file=f)
    print(path,b,d,bb,db)
    #exit()
    #"""
    fig8=plt.figure(1+i)
    plt.subplot(211)
    #plt.scatter(zenith_1,mag_tycho-mago_1,c=mago_1,cmap='viridis',s=3,alpha=0.8)
    plt.scatter(zenith_2,Rmag2-mag_v,c=mag_v,cmap='viridis',s=3,alpha=0.8)
    plt.colorbar(label='magnitude')
    plt.xlabel('zenith')
    plt.ylabel('Vmag(O-C)')
    plt.plot([65,65],[-2,4],'k--')
    plt.ylim(-2,4)
    plt.xlim(0,90)
    plt.title(path)

    k=0.35
    c0=np.array([x for x in range(0,90)],dtype=float)
    plt.plot(c0,t2fit(c0,b,d),'k-',label='fitting') # 4
    plt.plot(c0,d+k-k/np.cos(c0*np.pi/180),'r-',label='airmass k=%.01f'%k)
    #plt.plot(c0,t3fit(c0,a,b,c,d,e),'k-',label='fitting')
    plt.ylim(-2,4)
    plt.legend()
    plt.subplot(212)
    #plt.scatter(zenith_1,magoc-t2fit(zenith_1,b1,d1),c=mag_1+t2fit(zenith_1,b1,d1),cmap='viridis',s=3,alpha=0.8,label='num=%i, back=%i,a=%.04f, b=%.02f'%(num_p,np.median(back_p),b,d))    # 5
    plt.scatter(zenith_2,Rmag2-mag_v2,c=mag_v2,cmap='viridis',s=3,alpha=0.8,label='num=%i, back=%i,a=%.04f, b=%.02f'%(num_p,np.median(back_p),b,d))    # 5
    plt.colorbar()
    #plt.plot([0,90],[0,0],'k-',label='std=%.03f clipped=%.03f'%(np.std(magoc-t2fit(zenith_1,b1,d1)),std1))
    plt.plot([0,90],[0,0],'k-',label='std=%.03f'%(np.std(Rmag2-mag_v2)))
    plt.plot([65,65],[-2,4],'k--')
    plt.xlabel('zenith')
    plt.ylabel('Vmag(O-C)')
    plt.ylim(-2,4)
    plt.xlim(0,90)
    plt.legend()
    i=i+1
    
    plt.subplots_adjust(hspace=0)
    #plt.show()
    #plt.savefig('figure_fit/figure_fit_colorcorr/'+path+'.png')
    #plt.close()
    #"""

    # imcheck
    #"""
    DELTA=Rmag2-mag_v2
    MAG,DLT=[[],[]]
    for i in range(28):
        j=i*0.25
        k=j+0.25
        MAG.append(j+0.125)
        #index=(mag1>j)&(mag1<=k)
        index=(mag_v2>j)&(mag_v2<=k)                      #3
        dlt=np.std(DELTA[index])
        DLT.append(dlt)
    fig9=plt.figure(145+i)
    plt.subplot(211)
    plt.scatter(Rmag2,Rmag2-mag_v2,s=3)
    #plt.colorbar()
    plt.xlabel('Gmag')
    plt.ylabel('Vmag-Gmag')
    plt.xlim(0,7)
    plt.ylim(-1,1)
    #plt.title(path)

    plt.subplot(212)
    plt.plot(MAG,DLT,'k.',label='d=6')
    plt.plot([0,7],[0.1,0.1])
    plt.xlabel('Gmag')
    plt.ylabel('sigma mag')
    plt.xlim(0,7)
    plt.ylim(-0.1,0.41)
    plt.legend()
    
    plt.show()

    #fig9=plt.figure(145+i)
    #plt.plot(zenith_1,back,'r.',label='background')
    #plt.xlabel('zenith')
    #plt.ylabel('background (ADU)')
    #plt.title(path)
    #plt.savefig('figure_back/'+path+'.png')

    #fig9=plt.figure(145+i)
    #plt.plot(zenith_1,back,'r.',label='background')
    #plt.hist(back_p,range=(0,500),bins=100)
    #plt.plot([np.median(back_p),np.median(back_p)],[0,120],'r-',label='median=%i'%np.median(back_p))
    #plt.xlabel('zenith')
    #plt.ylabel('background (ADU)')
    #plt.title(path)
    #plt.legend()
    #plt.savefig('figure_back/'+path+'.png')
    #"""
    
#plt.show()
#f.close()
#"""

"""
# plot
mag_101,mag_tycho,zenith_1,x_1,y_1,ra_1,dec_1,Rra_1,Rdec_1,magoc,a,b,c,d,e=single()

plt.figure(1)
plt.subplot(211)
plt.title('sextractor')
plt.plot(zenith_1,mag_tycho-mag_101,'r.')
plt.ylim(-2.5,2.5)
plt.xlim(0,90)
plt.subplot(212)
plt.plot(mag_tycho,mag_tycho-mag_101,'r.')
plt.ylim(-3,3)
plt.xlim(0,8)
    

fig8=plt.figure(3)
plt.subplot(211)
plt.plot(zenith_1,magoc,'r.')
plt.xlabel('zenith')
plt.ylabel('Vmag(O-C)')



c0=np.array([x for x in range(0,90)],dtype=float)
plt.plot(c0,t3fit(c0,a,b,c,d,e),'b-',label='fitting')
plt.ylim(-2,4)
plt.legend()
plt.subplot(212)
plt.plot(zenith_1,magoc-t3fit(zenith_1,a,b,c,d,e),'r.',label='fitting')
plt.plot([0,90],[0,0],'b-')
plt.xlabel('zenith')
plt.ylabel('Vmag(O-C)')
plt.ylim(-2,4)
plt.legend()


plt.figure(5)
mag_101new=mag_101+t3fit(zenith_1,a,b,c,d,e)

index_m=np.where(zenith_1<10)[0]
x_m=x_1[index_m]
y_m=y_1[index_m]
mag_101m=mag_101[index_m]
mag_101nm=mag_101new[index_m]
mag_tychom=mag_tycho[index_m]
Rra_m=Rra_1[index_m]
Rdec_m=Rdec_1[index_m]

#for i in range(len(x_m)):
#    print(x_m[i],y_m[i],mag_101m[i],mag_101nm[i],mag_tychom[i],Rra_m[i],Rdec_m[i])



plt.subplot(211)
plt.title('sextractor_corrected')
plt.plot(zenith_1,mag_tycho-mag_101new,'r.')
plt.ylim(-2.5,2.5)
plt.xlim(0,90)

plt.subplot(212)
plt.plot(mag_tycho,mag_tycho-mag_101new,'r.')
plt.ylim(-3,3)
plt.xlim(0,8)
#"""



"""
# compare

mag2_101,mag2_tycho,zenith_2,x_2,y_2,ra_2,dec_2,Rra2_1,Rdec2_1,magoc2,a2,b2,c2,d2,e2=single('cali_back16/KLCAM_20170620003016_1.cr2_aper.cat')

sep1,sep2=crossmatch(ra_1,dec_1,ra_2,dec_2,4)

mag1=mag_101[sep1]
zenith1=zenith_1[sep1]
mag2=mag2_101[sep2]
zenith2=zenith_2[sep2]

zp1=t3fit(zenith1,a,b,c,d,e)
zp2=t3fit(zenith2,a2,b2,c2,d2,e2)

mag1new=mag1+zp1
mag2new=mag2+zp2
DELTA=mag1new-mag2new
MAG=[]
DLT=[]
for i in range(28):
    j=i*0.25
    k=j+0.25
    MAG.append(j+0.125)
    index=(mag1new>j)&(mag1new<=k)
    dlt=np.std(DELTA[index])
    #print(dlt)
    DLT.append(dlt)
plt.figure(7)
plt.subplot(211)
plt.plot(mag1new,mag1new-mag2new,'k.',markersize=1)
plt.xlabel('magnitude')
plt.ylabel('delta mag')
plt.ylim(-1,1)
plt.xlim(0,7)
plt.subplot(212)
plt.plot(MAG,DLT,'k.',label='d=10')
plt.xlabel('magnitude')
plt.ylabel('sigma mag')
plt.legend()
plt.xlim(0,7)
plt.ylim(-0.1,1.1)
plt.grid()


def ds9output(x,y,name,color='green'):
    f=open(name,'w+')
    print('# Region file format: DS9 version 4.1',file=f)
    print('global color='+color+' dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',file=f)
    print('physical',file=f)
    for i in range(len(x)):
        print('circle(%.02f,%.02f,10)'%(x[i],y[i]),file=f)
    f.close()
#"""

"""
mag_1new=mag_101+t3fit(zenith_1,a,b,c,d,e)
index1=set(list(np.where(mag_1new>2.9)[0]))
index2=set(list(np.where(mag_1new<2.7)[0]))
index_m=np.array(list(index1|index2))
x_m=np.delete(x_1,index_m)
y_m=np.delete(y_1,index_m)
mag_101m=np.delete(mag_101,index_m)
mag_101nm=np.delete(mag_101+t3fit(zenith_1,a,b,c,d,e),index_m)
mag_tychom=np.delete(mag_tycho,index_m)
zenith_m=np.delete(zenith_1,index_m)
ra_m=np.delete(ra_1,index_m)
dec_m=np.delete(dec_1,index_m)
Rra_m=np.delete(Rra_1,index_m)
Rdec_m=np.delete(Rdec_1,index_m)

#ds9output(x_m,y_m,'KLCAM_20170620000016_1.cr2_aper.cat.reg','green')
for i in range(len(x_m)):
    print(x_m[i],y_m[i],mag_101nm[i]-mag_101m[i],mag_101nm[i],mag_tychom[i],Rra_m[i],Rdec_m[i],zenith_m[i])
#"""



