import numpy as np
import matplotlib.pyplot as plt
import scipy
from datetime import datetime,timedelta

name,iso0,num0,B0,ext0=np.loadtxt('time_vs_isonumzp.txt',unpack=True,dtype=str)

time0=name
time_s,time_u=[[],[]]
iso=np.array(iso0,dtype=float)
num=np.array(num0,dtype=int)
ext=np.array(ext0,dtype=float)
B=np.array(B0,dtype=float)
for i in range(len(time0)):
    time1=datetime.strptime(time0[i][6:20],'%Y%m%d%H%M%S')
    time=time1+timedelta(hours=8)
    time_s.append(time)
    time_u.append(time1.timestamp())
time_s=np.array(time_s)
time_u=np.array(time_u)


## fitting ##
#index=np.where((ext>2)|(ext<1.5))[0]
# for G band
#index1=set(list(np.where((iso==400)&(num<600))[0]))
#index2=set(list(np.where((iso==1600)&(num<900))[0]))
# for B band
index1=set(list(np.where((iso==400)&(num<500))[0]))
index2=set(list(np.where((iso==1600)&(num<700))[0]))
#index3=set(list(np.where(ext>2.0)[0]))
index=np.array(list(index1|index2))
ext_f=np.delete(ext,index)
name_f=np.delete(name,index)
time_uf=np.delete(time_u,index)
time_sf=np.delete(time_s,index)

coef=np.polyfit(time_uf-time_uf[0],ext_f,2)
a,b,c=coef
p=np.poly1d([a,b,c+0.13])

print(coef)
#f=open('../param_timecorr_B_2018.txt','w+')
#print(a,b,c,file=f)
#f.close()

## new zp ##
p_out=np.poly1d([a,b,0])
time_output=time_u-time_u[0]
zp_new=ext-p_out(time_output)

time_standard=datetime.strptime('20170505200002','%Y%m%d%H%M%S').timestamp()
time_initial=time_u[0]
c_out=-1*p_out(time_standard-time_initial)
p_out2=np.poly1d([a,b,c_out])
print(p_out(time_standard-time_initial),p_out(1),p_out2(time_standard-time_initial))

# save txt
#f=open('../statslist_photutils_B_timecorr.txt','w+')
#for i in range(len(zp_new)):
#    print(name[i],num[i],B[i],zp_new[i],file=f)
#f.close()

## check im###
import matplotlib
matplotlib.rcParams['font.size'] = 16
plt.figure(1)
plt.plot(time_sf,ext_f,'r.',label='best images')
y_fit=p(time_uf-time_uf[0])
plt.gca().invert_yaxis()
plt.plot(time_sf,y_fit,'k-')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
plt.xlabel('Date')
plt.ylabel('zeropoint')
plt.title('year 2017')
plt.legend()

plt.figure(2)
index=np.where(num<100)[0]
time_plot=np.delete(time_s,index)
ext_plot=np.delete(ext,index)
zp_plot=np.delete(zp_new,index)

plt.plot(time_plot,ext_plot,'r.',label='all images')

y_fit=p(time_u-time_u[0])
plt.plot(time_s,y_fit,'k-',label='fitting from best images')

plt.xlabel('Date')
plt.ylabel('zeropoint')
plt.title('year 2017')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure(3)
plt.plot(time_s,zp_new,'r.',label='corrected')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
plt.xlabel('Date')
plt.ylabel('zeropoint')
plt.title('year 2017')
plt.legend()
plt.grid()

plt.figure(4)
plt.subplot(121)
index=np.where(num<100)[0]
time_plot=np.delete(time_s,index)
ext_plot=np.delete(ext,index)
plt.plot(time_plot,ext_plot,'r.',label='all images')
y_fit=p(time_u-time_u[0])
plt.plot(time_s,y_fit,'k-',label='fitting from best images')
plt.xlabel('Date')
plt.ylabel('zeropoint')
plt.title('year 2017')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
plt.grid()
plt.legend()

plt.subplot(122)
plt.plot(time_plot,zp_plot,'r.',label='corrected')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
plt.xlabel('Date')
plt.ylabel('zeropoint')
plt.title('year 2017')
plt.legend()
plt.ylim()
plt.grid()

plt.tight_layout()
plt.show()



