import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry,ApertureStats,CircularAnnulus
from astropy.stats import SigmaClip,sigma_clipped_stats
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve
#from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceFinder, SourceCatalog
from multiprocessing import Pool, cpu_count
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
import time
sigclip = SigmaClip(sigma=3, maxiters=5)

###### background estimation ######
@njit
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
@njit(parallel=True)
def _fast_mask(x0, y0, r_in, r_out, img_shape):
    mask = np.zeros(img_shape, dtype=np.bool_)
    for i in prange(max(0, int(y0-r_out)-1), min(img_shape[0], int(y0+r_out)+2)):
        for j in prange(max(0, int(x0-r_out)-1), min(img_shape[1], int(x0+r_out)+2)):
            dist_sq = (i-y0)**2 + (j-x0)**2
            if r_in**2 <= dist_sq <= r_out**2:
                mask[i,j] = True
    return mask


def _batch_fit(points, data, r_in, r_out):
    #r_sq_in, r_sq_out = r_in**2, r_out**2
    results = np.empty(len(points))
    
    for k in prange(len(points)):
        x0, y0 = points[k]
        #print(data.shape)
        #exit()
        mask = _fast_mask(x0, y0, r_in, r_out, data.shape)
        region = data[mask]
    
        if len(region) > 10:
            #left=min(region)
            #right=max(region)
            #len_bins=int((right-left)/5)
            # plt
            #hist,bins,patches=plt.hist(region,density=True,bins=len_bins)
            #plt.close()
            # numpy
            region=np.delete(region,sigclip(region).mask)
            hist, bins = np.histogram(region, bins='auto', density=True)# bins can be 'auto'
            bin_centers = (bins[:-1] + bins[1:]) / 2
            try:
                popt, _ = curve_fit(gaussian, bin_centers, hist, 
                                    p0=[1, np.mean(region), np.std(region)],
                                    maxfev=200)
                results[k] = popt[1]
                continue
            except:
                #print(len(region),x0,y0)
                pass
        #print(points)
        #print(len(region),x0,y0)
        results[k] = np.median(region)
    return results

def fast_photometry(positions, r_in, r_out, data, n_workers=8):
    points = np.column_stack((positions[:,0], positions[:,1]))
    chunk_size = len(points) // (n_workers * 4)
    #print(len(points),chunk_size)
    if(chunk_size==0):chunk_size=1
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(0, len(points), chunk_size):
            chunk = points[i:i+chunk_size]
            futures.append(executor.submit(_batch_fit, chunk, data, r_in, r_out))
        
        results = np.concatenate([f.result() for f in futures])
    
    return results
###################################

###### star finder ######
def circle_mask(x,y,x_c,y_c,r,mode='out'):
    Y,X = np.ogrid[:x,:y]
    dist = np.sqrt((X-x_c)**2+(Y-y_c)**2)
    if(mode=='out'):return dist>r
    else:return dist<=r

def tdback(data,s1=50,s2=3,mask=None):
    bkgest=MedianBackground()
    bkg=Background2D(data,(s1,s1),filter_size=(s2,s2),sigma_clip=sigclip,bkg_estimator=bkgest,coverage_mask=mask)
    return bkg

def starfind(path,x_c=1338,y_c=910,r_c=700):
    #print(path)
    with fits.open(path) as hdul:
        data=hdul[0].data

        mask=circle_mask(data.shape[0],data.shape[1],x_c,y_c,r_c)
        mask[841:970,1775:2040]=True

        mask1=circle_mask(data.shape[0],data.shape[1],x_c,y_c,r_c)
        bkg=tdback(data,s1=15,s2=1,mask=mask1)
        #plt.imshow(data-bkg.background,origin='lower', cmap='Greys_r',interpolation='nearest')
        #plt.show()
        
        
        threshold = 1.5 * bkg.background_rms

        kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        data_s=data-bkg.background
        convolved_data = convolve(data_s, kernel, mask=mask1)
        
        segment_map = detect_sources(convolved_data, threshold, npixels=4, mask=mask1)
        #print(segment_map)
        
        segm_deblend = deblend_sources(convolved_data, segment_map,npixels=4, nlevels=32, contrast=0.001,progress_bar=False)
        cat = SourceCatalog(data_s, segm_deblend,convolved_data=convolved_data, mask=mask,background=bkg.background)
        #back=cat.background_sum
        #print(len(back))
        # by testing, finder is not as good as seperation
        # finder = SourceFinder(npixels=4, progress_bar=False)        
        # segment_map = finder(convolved_data, threshold, mask=mask1)

        #cat = SourceCatalog(data_s, segment_map, convolved_data=convolved_data, mask=mask)
        
        sources=cat.to_table()
        
        
        sources['xcentroid'].info.format = '.2f'  # optional format
        sources['ycentroid'].info.format = '.2f'
        sources['kron_flux'].info.format = '.2f'
        
        return sources
####################################

###### photometry ######
def photometry(target,positions1=0,iso=1600.0,back_ini=[],r_in=6,r_out=12,buff=1,backdif=0):
    if(positions1==0):
        positions=starfind('/media/yangxu/disk4/klcamfits/'+target)
        x=positions['xcentroid'].data
        y=positions['ycentroid'].data
        flux_k=positions['kron_flux'].data
        flux_k=flux_k[~np.isnan(x)]
        y=y[~np.isnan(x)]
        x=x[~np.isnan(x)]        
	#print(x,y)
        positions1=np.array([x,y]).T
        #print(positions1)
        #print(len(x))
        if(len(x)==0):
            return []
    radii=[1.0,2.0,3.0,4.0,5.0]
    apertures1 = [CircularAperture(positions1, r=r) for r in radii]
    #annuluss1 = [CircularAnnulus(positions1, r_in=r_in, r_out=r_out) for r in radii]
    #aperture1 = CircularAperture(positions1, r=3)
    #annulus1 = CircularAnnulus(positions1, r_in=r_in, r_out=r_out)
    #print(len(apertures1))
    zp_flux=63095.7315
    iso_corr=1600.0/iso
    with fits.open('/media/yangxu/disk4/klcamfits/'+target) as hdul:
        data=hdul[0].data
        back=fast_photometry(positions1,r_in,r_out,data)
        #back=np.median(data)
        flux=[]
        mag=[]
        fwhm=[]
        for i in range(0,5):
            apstat=ApertureStats(data,apertures1[i])#.to_table()
            area=apstat.sum_aper_area.value
            aper_backsub=(apstat.sum-back*area)*iso_corr
            flux.append(aper_backsub)
            mag.append(-2.5*np.log10(aper_backsub/zp_flux))
            fwhm.append(apstat.fwhm)
        flux_k=flux_k*iso_corr
        mag_k=-2.5*np.log10(flux_k/zp_flux)
    #print(area)
    data_out=[x+1,y+1,mag[0],mag[1],mag[2],mag[3],mag[4],flux[0],flux[1],flux[2],flux[3],flux[4],mag_k,flux_k,back,fwhm[2]]
    hdul_table=Table(data_out,names=('x','y','mag2','mag4','mag6','mag8','mag10','flux2','flux4','flux6','flux8','flux10','mag_kron','flux_kron','back','fwhm'))
    return hdul_table

START=time.time()
lists,ISO=np.loadtxt('nightiso.txt',unpack=True,dtype=str)
for i in range(1442,len(lists)):
#for i in range(1441,1442):
    target=lists[i]
    iso=float(ISO[i])
    name,_=target.split('.')
    color_name=['Green1.fits','Blue.fits','Red.fits']
    color=['g','b','r']
    for j in range(0,3):
        starttime=time.time()
        A=photometry(target+color_name[j],iso=iso)
        endtime=time.time()
        A.write('klcam/catalog_photutils/'+target+color_name[j]+'.'+color[j]+'.cat',format='fits',overwrite=True)
#        print(target,'done')
        print(target,'%is'%(endtime-starttime),'%i/%i'%(i+1,len(lists)))
END=time.time()
print('total time: %is'%(END-START))
