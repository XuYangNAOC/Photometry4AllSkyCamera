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

def crossmatch(ra1,dec1,ra2,dec2,maxstep):
    wcs1=SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree)
    wcs2=SkyCoord(ra=ra2*u.degree,dec=dec2*u.degree)
    idx,d2d,d3d=wcs1.match_to_catalog_3d(wcs2)
    max_sep=maxstep*u.arcmin
    sep_c=d2d<max_sep
    return sep_c,idx[sep_c]

def single(path1='cali_back16/KLCAM_20170620000016_1.cr2_aper.cat',
           #path2='cali_back16/KLCAM_20170620003016_1.cr2_aper.cat',
           mode='test'):
    
    Rra,Rdec,Rmag,RBmag=np.loadtxt('cali_colour/tycho_v6_BV_fullsky.txt',skiprows=1,dtype=float,unpack=True)

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
    back1=data1[20]
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
    #"""
    #if not, return error. should be written in a function later
    #index1=set(list(np.where(fwhm_1<1)[0]))
    #index2=set(list(np.where(fwhm_1>10)[0]))
    #index3=set(list(np.where(mag_101>10)[0]))
    #index4=set(list(np.where(mag_101<-10)[0]))
    #index5=set(list(np.where(zenith_1>65)[0]))
    #index=np.array(list(index1|index2|index3|index4|index5))
    #print(len(index))
    index=np.where(zenith_1>90)[0]
    mag_101=np.delete(mag_101,index)
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
    #popt,pcov=curve_fit(t3fit,zenith_1,magoc)
    #a,b,c,d,e=popt
    popt,pcov=curve_fit(t2fit,zenith_1,magoc)
    b,d=popt                                    # 1

    #"""
    #mag_101new=mag_101+t3fit(zenith_1,a,b,c,d,e)
    mag_101new=mag_101+t2fit(zenith_1,b,d)      # 2
    magoc_new=mag_tycho-mag_101new
    magoc_clip=sigclip(magoc_new)
    #print(mag_101clip)
    index=np.where(magoc_new==magoc_clip)[0]
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
    if(mode=='test'):
        return num,mag_101,mag_tycho,back_p,azimuth_1,zenith_1,x_1,y_1,ra_1,dec_1,Rra_1,Rdec_1,magoc,a,b,c,d,e,std

#"""
# main loop
#f=open('statslist_withoutcolor.txt','w+')
#lists=np.loadtxt('colourlist.txt',unpack=True,dtype='str')
#lists=np.loadtxt('nightlist.txt',unpack=True,dtype='str')
lists=['KLCAM_20170505200002_1.g.cat']
i=0
for path in lists:
    name,color,_=path.split('.')
    if(color!='g'):continue
    rr=single(path1='fitback/'+name+'.backfit.g.cat')
    #rr=single(path1='cali_back16/'+path)
    #color=path.split('.')[1]
    if(rr==0):
        continue
    num,mag_101,mag_tycho,back,azimuth_1,zenith_1,x_1,y_1,ra_1,dec_1,Rra_1,Rdec_1,magoc,a,b,c,d,e,std=rr
    magoc=mag_tycho-mag_101

    back_p=back
    num_p=len(back_p)
    print('%.04f, %.04f, %.04f, %.04f, %.04f'%(a,b,c,d,e))
    #print(path,num_p,np.median(back_p),a,b,c,d,e,file=f)

    #"""
    fig8=plt.figure(1+i)
    plt.subplot(211)
    plt.scatter(zenith_1,magoc,s=3,alpha=0.8,color='g',label='before correction')
    #plt.scatter(zenith_1,magoc,c=mag_101,cmap='viridis',s=3,alpha=0.8)
    #plt.colorbar()
    #plt.xlabel('zenith')
    plt.ylabel(r'$V - G$ (magnitude)')
    plt.plot([65,65],[-2,4],'k--')
    
    plt.xlim(0,90)
    plt.title(path[:-6])

    k=0.3
    c0=np.array([x for x in range(0,90)],dtype=float)
    plt.plot(c0,t2fit(c0,b,d),'k-',label='2nd-order fitting') # 4
    #plt.plot(c0,d+k-k/np.cos(c0*np.pi/180),'r-',label='airmass k=%.01f'%k)
    #plt.plot(c0,t3fit(c0,a,b,c,d,e),'k-',label='fitting')
    plt.ylim(-2.1,3.9)
    plt.legend()
    plt.xticks([])
    
    plt.subplot(212)
    plt.scatter(zenith_1,magoc-t2fit(zenith_1,b,d),s=3,alpha=0.8,color='g',label='after correction')
    #plt.scatter(zenith_1,magoc-t2fit(zenith_1,b,d),c=mag_101+t2fit(zenith_1,b,d),cmap='viridis',s=3,alpha=0.8,label='num=%i, back=%i,a=%.04f, b=%.02f'%(num_p,np.median(back_p),b,d))    # 5
    #plt.scatter(zenith_1,magoc-t3fit(zenith_1,a,b,c,d,e),c=azimuth_1,cmap='viridis',s=3,alpha=0.8,label='num=%i, back=%i,a=%.04f, b=%.02f'%(num_p,np.median(back_p),b,d))    # 5
    #plt.colorbar()
    #plt.plot([0,90],[0,0],'k-',label='std=%.03f clipped=%.03f'%(np.std(magoc-t2fit(zenith_1,b,d)),std))
    plt.plot([0,90],[0,0],'k-')#,label='std=%.03f'%(std))
    plt.plot([65,65],[-2,4],'k--')
    plt.xlabel(r'zenith distance (degree)')
    plt.ylabel(r'$V - G$ (magnitude)')
    plt.ylim(-2.1,3.9)
    plt.xlim(0,90)
    plt.legend(loc=2)
    i=i+1
    
    plt.subplots_adjust(hspace=0)
    plt.show()
    #plt.savefig('figure_fit/figure_fit_2order/'+path+'.png')
    #plt.savefig('figure_colour/'+path+'.png')
    plt.close()
    #fig9=plt.figure(145+i)
    #plt.subplot(211)
    #plt.scatter(azimuth_1,magoc,c=zenith_1,cmap='viridis',s=3,alpha=0.8)
    #plt.colorbar()
    #plt.xlabel('azimuth')
    #plt.ylabel('Vmag(O-C)')
    #plt.title(path)

    #plt.subplot(212)
    #plt.scatter(azimuth_1,magoc-t2fit(zenith_1,b,d),c=zenith_1,cmap='viridis',s=3,alpha=0.8,label='num=%i, back=%i,a=%.04f, b=%.02f'%(num_p,np.median(back_p),b,d))    # 5
    #plt.scatter(azimuth_1,magoc-t3fit(zenith_1,a,b,c,d,e),c=zenith_1,cmap='viridis',s=3,alpha=0.8,label='num=%i, back=%i,a=%.04f, b=%.02f'%(num_p,np.median(back_p),b,d))    # 5
    #plt.colorbar()
    #plt.xlabel('azimuth')
    #plt.ylabel('Vmag(O-C)')
    #plt.legend()
    #plt.savefig('figure_back/'+path+'.png'

    #plt.show()

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



