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
from datetime import datetime

import time
starttime=time.time()
sigclip = SigmaClip(sigma=3.0, maxiters=10)

#def t3fit(x,a,b,c,d,e):
def t3fit(x,b,d):
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

def solver(k1,k2,k3,k4,y):
    poly_coeffs=[k4,0,k3,0,k2,0,k1,0]
    def solve_for_y(poly_coeffs, y):
        pc = poly_coeffs.copy()
        pc[-1] -= y
        return np.roots(pc)
    R=solve_for_y(poly_coeffs,y)
    buffs=[]
    for i in R:
        if(i.imag==0):
            r=i.real
            return r
        else:continue
        print('solution not exist')
        return 0

def radectoub(Rra,Rdec,x0,y0,E,a0,e,k1,k2,k3,k4,time0,loc):
    # not in the pipeline, but useful
    time = Time(time0)
    cord = SkyCoord(ra=Rra*u.degree,dec=Rdec*u.degree)
    Raltaz=cord.transform_to(AltAz(obstime=time,location=loc))
    z=np.pi/2-Raltaz.alt.rad
    a=Raltaz.az.rad
    u0=np.arccos(np.cos(z)*np.cos(e)+np.sin(z)*np.sin(e)*np.cos(a-E))
    b0=np.arctan2(np.sin(a-E)*np.sin(z)/np.sin(u0),-1*(np.cos(z)-np.cos(u0)*np.cos(e))/(np.sin(u0)*np.sin(e)))
    
    ca=np.array([x-a0+E if x-a0+E>0 else x-a0+E+2*np.pi for x in b0])
    R=[]
    for i in range(len(u0)):
        r=solver(k1,k2,k3,k4,u0[i])
        R.append(r)
    R=np.array(R)
    x=x0-R*np.sin(ca)
    y=y0-R*np.cos(ca)
    return x,y,90-Raltaz.alt.degree,Raltaz.az.degree

def ds9output(x,y,name,color='green',size=10):
    f=open(name+'.reg','w+')
    print('# Region file format: DS9 version 4.1',file=f)
    print('global color='+color+' dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',file=f)
    print('physical',file=f)
    for i in range(len(x)):
        print('circle(%.02f,%.02f,%i)'%(x[i],y[i],size),file=f)
    f.close()
    
def reading(path1='cali_back16/KLCAM_20170620000016_1.cr2_aper.cat',
           #path2='cali_back16/KLCAM_20170620003016_1.cr2_aper.cat',
           Ca=0,
           Cb=0,
           mode='test'):
    
    #Rra,Rdec,Rmag,RBmag=np.loadtxt('standard_catalogue/tycho_v6_novary.v2.txt',skiprows=1,dtype=float,unpack=True)

    data1=np.loadtxt(path1,unpack=True)

    x1,y1=data1[0:2]
    ra1,dec1=data1[2:4]
    mag21,mag41,mag61,mag81,mag101=data1[4:9]
    mag_kron=data1[9]
    flux_kron=data1[10]
    fwhm1=data1[11]
    flux21,flux41,flux61,flux81,flux101=data1[12:17]
    azimuth1=data1[17]
    zenith1=data1[18]
    back1=data1[19]
    
    return mag21,mag41,mag61,mag81,mag101,ra1,dec1,x1,y1,azimuth1,zenith1

def colorcoef(name,coef,mode='test'):
    b,d=coef
    tar=name

    rr_b=reading(path1='../cali_photutils/'+tar+'.g.cat',mode='65')
    rr_g=reading(path1='../cali_photutils/'+tar+'.g.cat',mode='65')
    mag21,mag41,mag61,mag81,mag101,ra1,dec1,x1,y1,azimuth1,zenith1=rr_b
    mag22,mag42,mag62,mag82,mag102,ra2,dec2,x2,y2,azimuth2,zenith2=rr_g

    wcs1=SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree)
    wcs2=SkyCoord(ra=ra2*u.degree,dec=dec2*u.degree)

    #idx,d2d,d3d=wcs1.match_to_catalog_3d(wcs2)
    #max_sep=21*u.arcmin
    #sep_c=d2d<max_sep
    
    #mag1=mag61[sep_c]+t3fit(zenith1[sep_c],b1,d1)
    
    time0=datetime.strptime('20170329190002','%Y%m%d%H%M%S').timestamp()
    time2=datetime.strptime(name[6:20],'%Y%m%d%H%M%S').timestamp()
    time=time2-time0
    #print(time)
    a0,b0,c0=np.loadtxt('../param_timecorr.txt',dtype=float)
    p=np.poly1d([a0,b0,0])
    d=d+p(time)    # 文件夹extinction/里用的是加，结果是对的
    #d=d-p(time)     # 文件夹extinction3/里用减，结果是错的
    #if(mode=='2'):
    #    d=d+p(time)
    #else:
        #d=d-p(time)
    mag2=mag62+t3fit(zenith2,b,d)
    if(mode=='blue'):
        mag2=mag61+t3fit(zenith1,b,d)
    #mag_v=(1-Ca)*mag1+Ca*mag2-Cb
    
    #ra_2=ra2[idx[sep_c]]
    #dec_2=dec2[idx[sep_c]]
    #az_2=azimuth2[idx[sep_c]]
    #ze_2=zenith2[idx[sep_c]]
    #x_2=x2[idx[sep_c]]
    #y_2=y2[idx[sep_c]]

    return mag2,x2,y2,ra2,dec2,azimuth2,zenith2,p(time)

def single(name1='KLCAM_20170505200002_1',
           name2='KLCAM_20170621210017_1',
           mode='test'):
    # b,d, given by the best image
    coefs=[-0.00011011102441934997,
           1.7518468667228
           ]
    coefs_b=[-0.000149841,
             1.98396307
             ]

    path1='../cali_photutils/'+name1+'.g.cat'
    path2='../cali_photutils/'+name2+'.g.cat'
    mag1,x1,y1,ra1,dec1,az1,ze1,p=colorcoef(name1,coefs)
    mag2,x2,y2,ra2,dec2,az2,ze2,p=colorcoef(name2,coefs)
    if(mode=='blue'):
        path1b='../cali_photutils/'+name1+'.b.cat'
        path2b='../cali_photutils/'+name2+'.b.cat'
        mag1,x1,y1,ra1,dec1,az1,ze1,p=colorcoef(name1,coefs,mode=mode)
        mag2,x2,y2,ra2,dec2,az2,ze2,p=colorcoef(name2,coefs,mode=mode)
    # path1 as standard catalogue
    #Rra,Rdec,Rmag,RBmag=np.loadtxt('standard_catalogue/tycho_v6_novary.txt',skiprows=1,dtype=float,unpack=True)

    
    wcs1=SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree)
    wcs2=SkyCoord(ra=ra2*u.degree,dec=dec2*u.degree)

    # time
    name=path2.split('/')[-1]
    time0=name.split('_')[1]
    time1=time0[0:4]+'-'+time0[4:6]+'-'+time0[6:8]+' '+time0[8:10]+':'+time0[10:12]+':'+time0[12:14]
    time2 = Time(time1)
    #time2 = Time('2017-06-21 23:30:17')
    domea = EarthLocation(lat=-80.41694*u.deg, lon=77.11611*u.deg, height=4093*u.m)
    x0,y0=np.loadtxt('x0y0.txt')
    #name2=path2.split('/')[-1]
    E,a0,e0,k1,k2,k3,k4=np.loadtxt('../astropara/'+name2.split('.')[0]+'.cr2_parameters.txt')

    index=np.where(mag1<=5.5)[0]
    c_ra=ra1[index]
    c_dec=dec1[index]
    c_az=az1[index]
    c_ze=ze1[index]
    c_mag=mag1[index]
    x_new,y_new,c_ze,c_az=radectoub(c_ra,c_dec,x0,y0,E,a0,e0,k1,k2,k3,k4,time2,domea)

    #check
    #ds9output(x_new,y_new,path2.split('/')[-1]+'_reference')
    #ds9output(x2,y2,path2.split('/')[-1]+'_ini')
    
    # compare


    idx,d2d,d3d=wcs1.match_to_catalog_3d(wcs2)
    max_sep=21*u.arcmin
    sep_c=d2d<max_sep

    ra_1=ra1[sep_c]
    dec_1=dec1[sep_c]
    x_1=x1[sep_c]
    y_1=y1[sep_c]
    mag_1=mag1[sep_c]
    zenith_1=ze1[sep_c]
    azimuth_1=az1[sep_c]

    
    mag_2=mag2[idx[sep_c]]
    azimuth_2=az2[idx[sep_c]]
    zenith_2=ze2[idx[sep_c]]
    x_2=x2[idx[sep_c]]
    y_2=y2[idx[sep_c]]

    #ds9output(x_2,y_2,path2.'ds9/'+split('/')[-1]+'_crossed')

    magoc=mag_2-mag_1
    #magoc[magoc<0]=0
    #index1=set(list(np.where(mag_1>5.5)[0]))
    index2=set(list(np.where(zenith_2>65)[0]))
    index3=set(list(np.where(x_2>1762)[0]))&set(list(np.where(y_2>833)[0]))&set(list(np.where(y_2<960)[0]))
    #index3=set(list(np.where(magoc>4)[0]))
    index=np.array(list(index2|index3))
    #print(len(magoc),len(index))
    if(len(index)==0):
        return 0
    Rmagoc=np.delete(magoc,index)
    Rx=np.delete(x_2,index)
    Ry=np.delete(y_2,index)
    Rm1=np.delete(mag_1,index)
    Rm2=np.delete(mag_2,index)
    Rra=np.delete(ra_1,index)
    Rdec=np.delete(dec_1,index)
    Raz=np.delete(azimuth_2,index)
    Rze=np.delete(zenith_2,index)

    """
    #minx=np.where(Rmagoc<0)[0]
    # check
    minx=np.where(abs(Rmagoc-np.median(Rmagoc))>3*np.std(Rmagoc))[0]
    spx=Rx[minx]
    spy=Ry[minx]
    spoc=Rmagoc[minx]
    spm1=Rm1[minx]
    spm2=Rm2[minx]
    spaz=Raz[minx]
    spze=Rze[minx]
    
    
    # region
    S_alt=[0,44.7,65]
    REG=[]
    SUP=[]
    Rinfo=[]
    count=0
    
    for i in range(0,360,45):
        for j in range(0,2):
            az1=i
            az2=i+45
            ze1=S_alt[j]
            ze2=S_alt[j+1]
            index1=set(list(np.where(Raz<az1)[0]))
            index2=set(list(np.where(Raz>=az2)[0]))
            index3=set(list(np.where(Rze<ze1)[0]))
            index4=set(list(np.where(Rze>=ze2)[0]))
            index=np.array(list(index1|index2|index3|index4))
            region=np.delete(Rmagoc,index)
            REG.append(region)
            Rinfo.append([(az1,az2),(ze1,ze2)])

            index1=set(list(np.where(c_az<az1)[0]))
            index2=set(list(np.where(c_az>=az2)[0]))
            index3=set(list(np.where(c_ze<ze1)[0]))
            index4=set(list(np.where(c_ze>=ze2)[0]))
            index=np.array(list(index1|index2|index3|index4))
            num=np.delete(c_mag,index)
            SUP.append(len(num))
    """  
    
    return Rmagoc,Rx,Ry,Rm1,Rm2,Raz,Rze,Rra,Rdec,x_new,y_new,c_az,c_ze,p


#name='KLCAM_20170621030017_1'
#ext,x,y,m1,m2,ra,dec,az,ze,spx,spy,spm1,spm2,sext,REG,Rinfo,SUP=single(name2=name)
#np.savetxt('extinction/'+name+'_ext.txt',np.array([ext,x,y,m1,m2,az,ze,ra,dec]).T,fmt='%f')
#ds9output(x,y,'ds9/'+name+'_ext')
data=np.loadtxt('../statslist_photutils_G_timecorr.txt',unpack=True,dtype=str)
lists=data[0]
num=data[1]
P=[]
#for i in range(len(lists)):
for i in range(100,101):
    name=lists[i].split('.')[0]
    print(name)
    if(num[i]=='0')or(num[i]=='1'):
        #np.savetxt('extinction_b/'+name+'_ext.txt',np.zeros(9),fmt='%f')
        continue
    rr=single(name2=name)
    """
    try:
        rr=single(name2=name)
    except:
        np.savetxt('extinction/'+name+'_ext.txt',np.zeros(9),fmt='%f')
        continue
    """
    if(rr==0):
        np.savetxt('extinction_b/'+name+'_ext.txt',np.zeros(9),fmt='%f')
        continue
    ext,x,y,m1,m2,az,ze,ra,dec,x_new,y_new,c_az,c_ze,p=rr
    print(len(ext))
    print(name,np.median(ext))
    #print(name,p)
    #spnum=len(spx)
    #np.savetxt('extinction_b/'+name+'_ext.txt',np.array([ext,x,y,m1,m2,az,ze,ra,dec]).T,fmt='%f')
    #np.savetxt('extinction_b/'+name+'_sp.txt',np.array([x_new,y_new,c_az,c_ze]).T,fmt='%f')
    #ds9output(x,y,'ds9/'+name+'_ext')
endtime=time.time()
print(endtime-starttime)
"""
#ext,x,y=single()
#for i in range(len(spx)):
#    print(i,spx[i],spy[i],spm1[i],spm2[i])
count=0
for i in range(len(Rinfo)):
    print(Rinfo[i],'%.02f'%np.mean(REG[i]),'%i/%i'%(len(REG[i]),SUP[i]),'%.01f%%'%(len(REG[i])/SUP[i]*100))
    count=count+len(REG[i])
print(len(spx),len(x),np.mean(spm1),count,sum(SUP))
plt.figure(1)
plt.scatter(x,y,c=ext,cmap='viridis',alpha=0.8,edgecolors='k',s=25)
#plt.scatter(spx,spy,color='r')
cbar=plt.colorbar(label='extinction')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.figure(2)
plt.plot(m1,ext,'k.')
#plt.plot(spm1,sext,'r.',)
plt.plot([min(m1),max(m1)],[np.median(ext),np.median(ext)],'b-',label='median=%.02f'%np.median(ext))
plt.legend()
plt.xlim(-0.1,6)
plt.ylim(-0.1,2.5)
plt.xlabel('magnitude')
plt.ylabel('extinction')

plt.show()

"""
