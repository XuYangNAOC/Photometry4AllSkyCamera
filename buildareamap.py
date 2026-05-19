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


def ds9output(x,y,name,color='green'):
    f=open(name+'.reg','w+')
    print('# Region file format: DS9 version 4.1',file=f)
    print('global color='+color+' dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',file=f)
    print('physical',file=f)
    for i in range(len(x)):
        print('circle(%.02f,%.02f,10)'%(x[i],y[i]),file=f)
    f.close()


# find center
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
# example
#num=-1
#rdif=solver(k1,k2,k3,k4,e0)
#xc=x0+num*rdif*np.cos(np.pi*2-(E-a0)-np.pi/2)
#yc=y0+num*rdif*np.sin(np.pi*2-(E-a0)-np.pi/2)
    
def taylor(xdata,k1,k2,k3,k4):
    return k1*xdata+k2*(xdata**3)+k3*(xdata**5)+k4*(xdata**7)

def xytoradec(x,y,x0,y0,E,a0,e,k1,k2,k3,k4,time0):
    cz=taylor(np.sqrt((x-x0)**2+(y-y0)**2),k1,k2,k3,k4)
    ca=np.arctan2(x-x0,y-y0)+np.pi
    u0=cz
    X=a0-E
    b=X+ca
    Cze=np.arccos(np.cos(u0)*np.cos(e)-np.cos(b)*np.sin(u0)*np.sin(e))

    amE = np.arctan2(np.sin(b)*np.sin(u0),(np.cos(b)*np.sin(u0)*np.cos(e)+np.cos(u0)*np.sin(e)))
    az0=amE+E
    #Caz=np.array([x+E if x+E >0 else x+E+np.pi*2 for x in amE])
    Caz=np.where(az0<0,az0+2*np.pi,az0)
    
    c1=90-Cze*180/np.pi
    c2=Caz*180/np.pi
    time = Time(time0)
    domea = EarthLocation(lat=-80.41694*u.deg, lon=77.11611*u.deg, height=4093*u.m)
    c0=AltAz(obstime=time,location=domea,az=c2*u.degree,alt=c1*u.degree)
    c00=SkyCoord(c0)
    c000=c00.transform_to('icrs')
    ra=c000.ra.degree
    dec=c000.dec.degree

    return Cze*180/np.pi,Caz*180/np.pi,ra,dec


time0='2017-05-05 20:00:02'
E,a0,e0,k1,k2,k3,k4=np.loadtxt('d:/seafile/私人资料库/photutils/astropara/KLCAM_20170505200002_1.cr2_parameters.txt')
x0,y0=np.loadtxt('x0y0.txt',unpack=True)
#print(E)
with fits.open('../fits/KLCAM_20170505200002_1.cr2Green1.fits') as hdul:
    data=hdul[0].data
    sp=data.shape
    X=np.zeros(sp)
    Y0=np.zeros(sp).T
print(sp)
X_1=np.zeros(sp)
Y_0=np.zeros(sp).T
for i in range(len(X)):
    X[i]=np.arange(0,2640)
    X_1[i]=np.arange(1,2641)
for i in range(len(Y0)):
    Y0[i]=np.arange(0,1764)
    Y_0[i]=np.arange(1,1765)
Y=Y0.T
Y_1=Y_0.T
#print(X,Y)
zenith,azimuth,ra,dec=xytoradec(X,Y,x0,y0,E,a0,e0,k1,k2,k3,k4,time0)
#np.savetxt('zenith_map.txt',zenith,fmt='%.08f')
#np.savetxt('azimuth_map.txt',azimuth,fmt='%.08f')
#print(zenith)
mask=np.load('KLCAM_20170505200002_1.cr2_VbandBack.mask.npy')
plt.figure(1)
p_z=np.ma.masked_where(mask,zenith)
plt.imshow(p_z,origin='lower', cmap='RdYlBu',interpolation='nearest')
cb1=plt.colorbar()
cb1.set_label(r'$degree$')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(680,2010)
plt.ylim(240,1580)
#plt.ylim(640,1970)
#plt.xlim(240,1580)
plt.title('zenith')
plt.figure(2)
p_a=np.ma.masked_where(mask,azimuth)
plt.imshow(p_a,origin='lower', cmap='RdYlBu',interpolation='nearest')
cb1=plt.colorbar()
cb1.set_label(r'$degree$')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(680,2010)
plt.ylim(240,1580)
plt.title('azimuth')
#plt.show()

zenithx,azimuthx,rax,decx=xytoradec(X_1,Y,x0,y0,E,a0,e0,k1,k2,k3,k4,time0)
zenithy,azimuthy,ray,decy=xytoradec(X,Y_1,x0,y0,E,a0,e0,k1,k2,k3,k4,time0)

C1=AltAz(alt=(90-zenith)*u.degree,az=azimuth*u.degree)
C2=AltAz(alt=(90-zenithx)*u.degree,az=azimuthx*u.degree)
C3=AltAz(alt=(90-zenithy)*u.degree,az=azimuthy*u.degree)
DIST1=C1.separation(C2)
DIST2=C1.separation(C3)
area=DIST1.arcsec*DIST2.arcsec
#np.savetxt('area_map.txt',area,fmt='%.08f')

plt.figure(3)
p_ra=np.ma.masked_where(mask,ra)
plt.imshow(p_ra,origin='lower', cmap='RdYlBu',interpolation='nearest')
cb1=plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(680,2010)
plt.ylim(240,1580)
plt.title('ra')

plt.figure(4)
p_dec=np.ma.masked_where(mask,dec)
plt.imshow(p_dec,origin='lower', cmap='RdYlBu',interpolation='nearest')
cb1=plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(680,2010)
plt.ylim(240,1580)
plt.title('dec')
plt.show()
