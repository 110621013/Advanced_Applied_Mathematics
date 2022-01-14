#本作業是關於變分分析在熱力學反演中的應用。下面是動量方程，圖 1 描繪了Z=6 處的水平風場。
#假設我們已經使用雷達衍生的 3D 風場計算動量方程右側的 F、G 和 H（如圖 2 所示，Z=6），您的任務是：
#(1) 從助教那裡獲取包含 F、 G、 H 和 𝜃𝑣0 、 𝜃0 的文件。
#(2) 繪製F、G 和H 的字段，並確保您輸入的數據正確。
#(3) 使用課堂上介紹的變分方法，檢索每一層的壓力和溫度擾動場（𝜋′，𝜃𝑐′）。請顯示 Z=2、4、6、8 處的結果。
#請注意，使用我們在課堂上討論的方法，您只能從水平平均值中恢復擾動的偏差。
# 在本作業中，您可以練習如何實現 Neumann 邊界條件。這份作業的截止日期是2022/1/14，也就是本學期的最後一天。

import os
import numpy as np
import matplotlib.pyplot as plt
x_n, y_n, z_n = 31, 21, 11
dx, dy, dz = 1000, 1000, 250

# read hw6 data, 31*21*11*5(x,y,z,var), z:1~11
def read_hw6_data(z, parameter):
    parameter_list = ['ff', 'gg', 'hh', 'theta0', 'thetav0', 'ans_thetac']
    data = np.fromfile('homework6.bin', dtype='<f4', count=-1, sep='')
    init_lndex = x_n*y_n*z_n*parameter_list.index(parameter) + x_n*y_n*(z-1)
    return np.reshape(data[init_lndex:init_lndex+x_n*y_n], (y_n, x_n))
# plot parameter
def plot_parameter(plane_data, title, scale):
    plane_data_new = np.array(plane_data)
    assert id(plane_data) != id(plane_data_new)
    plane_data_new *= scale
    C = plt.contourf(plane_data_new) #, levels=[i for i in range(-10, 10, 2)]
    plt.clabel(C, fontsize=8, colors='black', inline=False)
    plt.title(title)
    plt.xticks(range(0, 31, 5))
    plt.yticks(range(0, 21, 5))
    plt.savefig('6-'+title)
    #plt.show()
    plt.close()
# Input pre, post, and d and output interpolation differential.
def median_interpolation(front, behind, d):
    return (behind-front)/(2*d)

def cal_pi(F, G, z_name):
    # if pi has been cal
    filepath = os.path.join('.', '6-'+z_name+'_pi.npy')
    if os.path.isfile(filepath):
        pi = np.load(os.path.join('.', '6-'+z_name+'_pi.npy'))
    else:
        # pi B.C. init
        pi = np.zeros((y_n, x_n), dtype=np.float64)
        for y in range(1, y_n-1):
            pi[y, 0] = -2*dx*F[y, 1] + pi[y, 2]
            pi[y, x_n-1] = 2*dx*F[y, x_n-2] - pi[y, x_n-3]
        for x in range(1, x_n-1):
            pi[0, x] = -2*dy*F[1, x] + pi[2, x]
            pi[y_n-1, x] = 2*dy*F[y_n-2, x] - pi[y_n-3, x]

        # SOR + B.C. renew
        count = 0
        eps = np.array([1.5], dtype=np.float64)[0]
        threshold = np.array([10**-3], dtype=np.float64)[0]
        while True:
            old_pi = np.array(pi) # call by values
            count += 1

            for y in range(1, y_n-1):
                for x in range(1, x_n-1):
                    #sor_result = 2*dx*dy*(median_interpolation(F[y, x-1], F[y, x+1], dx) + median_interpolation(G[y-1, x], G[y+1, x], dy)) + (pi[y+1, x]+pi[y-1, x]+pi[y, x+1]+pi[y, x-1])/4
                    sor_result = -0.25*dx*dy*(median_interpolation(F[y, x-1], F[y, x+1], dx) + median_interpolation(G[y-1, x], G[y+1, x], dy)) + (pi[y+1, x]+pi[y-1, x]+pi[y, x+1]+pi[y, x-1])/4
                    pi[y, x] += eps * (sor_result - pi[y, x])
                
            skip_flag = False
            max_rela_e = np.array([0.0], dtype=np.float64)[0]
            for y in range(1, y_n-1):
                for x in range(1, x_n-1):
                    if old_pi[y, x] == np.array([0.0], dtype=np.float64)[0]:
                        max_rela_e = 10*threshold # abs fail
                        skip_flag = True
                    if skip_flag == True:
                        break
                    max_rela_e = max(max_rela_e, abs((pi[y, x]-old_pi[y, x])/old_pi[y, x]))
                if skip_flag == True:
                    break
            ''' MSE
            skip_flag = False
            mse = np.array([0.0], dtype=np.float64)[0]
            for i in range(2, n-2):
                for j in range(2, n-2):
                    if old_lamb[i][j] == np.array([0.0], dtype=np.float64)[0]:
                        mse = 10*mse_threshold # abs fail
                        skip_flag = True
                    if skip_flag == True:
                        break
                    mse += (lamb[i][j]-old_lamb[i][j])**2
                if skip_flag == True:
                    break
            '''
            # B.C. renew
            for y in range(1, y_n-1):
                pi[y, 0] = -2*dx*F[y, 1] + pi[y, 2]
                pi[y, x_n-1] = 2*dx*F[y, x_n-2] - pi[y, x_n-3]
            for x in range(1, x_n-1):
                pi[0, x] = -2*dy*G[1, x] + pi[2, x]
                pi[y_n-1, x] = 2*dy*G[y_n-2, x] - pi[y_n-3, x]


            if skip_flag or count%100==0:
                print(count, skip_flag, max_rela_e)
            if max_rela_e < threshold:
                break
            #if count == 4000:
            #    break
            #if count % 1000 == 0:
            #    plt.contourf(pi)
            #    plt.colorbar()
            #    plt.show()
        # -c
        pi_avg = np.mean(pi[0:y_n, 0:x_n])
        print('pi_avg:', pi_avg)
        pi -= pi_avg
        np.save('6-'+z_name+'_pi.npy', pi)
    plt.contourf(pi)
    plt.colorbar()
    plt.title(z_name+" pi'-<pi'>")
    plt.xticks(range(0, 31, 5))
    plt.yticks(range(0, 21, 5))
    plt.savefig('6-'+z_name+'_pi')
    #plt.show()
    plt.close()

def cal_theta_c(H, theta_0, theta_v0, z_name):
    # if pi has been cal
    filepath = os.path.join('.', '6-'+z_name+'_theta_c.npy')
    if os.path.isfile(filepath):
        theta_c = np.load(os.path.join('.', '6-'+z_name+'_theta_c.npy'))
    else:
        assert z_name in ['z='+str(i) for i in range(2, 11)]
        middle_num = int(z_name.split('=')[-1])
        under = np.load('6-z='+str(middle_num-1)+'_pi.npy')
        up = np.load('6-z='+str(middle_num+1)+'_pi.npy')
        dpidz = median_interpolation(under, up, dz)
        
        theta_0_avg, theta_v0_avg = np.mean(theta_0), np.mean(theta_v0)
        g = 9.8
        H -= np.mean(H)
        theta_c = (dpidz-H)*theta_0_avg*theta_v0_avg/g
        np.save('6-'+z_name+'_theta_c.npy', theta_c)

    plt.contourf(theta_c)
    plt.colorbar()
    plt.title(z_name+" theta_c'-<theta_c'>")
    plt.xticks(range(0, 31, 5))
    plt.yticks(range(0, 21, 5))
    plt.savefig('6-'+z_name+'_theta_c')
    #plt.show()
    plt.close()
    

if __name__ == '__main__':
    for z in range(2, 10, 2):
        f = read_hw6_data(z=z, parameter='ff')
        g = read_hw6_data(z=z, parameter='gg')
        h = read_hw6_data(z=z, parameter='hh')
        theta_0 = read_hw6_data(z=z, parameter='theta0')
        theta_v0 = read_hw6_data(z=z, parameter='thetav0')
        ans_thetac = read_hw6_data(z=z, parameter='ans_thetac')
    
        plot_parameter(f, 'z={} F field(multiplied by {})'.format(str(z), str(10**5)), scale=10**5)
        plot_parameter(g, 'z={} G field(multiplied by {})'.format(str(z), str(10**5)), scale=10**5)
        plot_parameter(h, 'z={} H field(multiplied by {})'.format(str(z), str(10**4)), scale=10**4)
        plot_parameter(theta_0, 'z={} theta0 field'.format(str(z)), scale=1)
        plot_parameter(theta_v0, 'z={} thetav0 field'.format(str(z)), scale=1)
        plot_parameter(ans_thetac, 'z={} ans_thetac field'.format(str(z)), scale=1)
    
        cal_pi(f, g, z_name='z='+str(z))
        
        f_under = read_hw6_data(z=z-1, parameter='ff')
        g_under = read_hw6_data(z=z-1, parameter='gg')
        cal_pi(f_under, g_under, z_name='z='+str(z-1))
        f_up = read_hw6_data(z=z+1, parameter='ff')
        g_up = read_hw6_data(z=z+1, parameter='gg')
        cal_pi(f_up, g_up, z_name='z='+str(z+1))
        cal_theta_c(h, theta_0, theta_v0, z_name='z='+str(z))