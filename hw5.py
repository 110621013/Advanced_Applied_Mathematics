'''
二維風場數據位於 Arakawa A 網格點系統上，這意味著所有變量都位於同一網格點。 請執行以下操作：
(1) 檢查平均散度的大小。 您可以使用均方根幅度。
(2) 繪製風場。
(3) 使用“正確”方案對強約束下的風場進行變分調整，使最終風場變得完全不可壓縮。 請分別對您的計算機代碼使用“單精度”和“雙精度”重複此問題兩次。
(4) 調整後再次計算平均散度的大小。
(5) 繪製調整後的風場。
(6) 重複 (3) 到 (5)，但使用“錯誤”但更準確的方案。
附言
(1) 提交結果和計算機代碼。
(2) 注意開始計算散度的起點。
'''
import os
import numpy as np
import matplotlib.pyplot as plt
n = 201
# u, v : n*n
# div, lamb = n*n [usful:(n-4)(n-4)]

# read data
def read_data():
    f = np.loadtxt('data.txt')
    u, v = f[:201], f[201:]
    return u, v
# thinning data for beautiful wind quiver
def thinning(field, thinning_times=5):
    new_field = np.zeros((n//thinning_times, n//thinning_times))
    for i in range(n):
        if i%thinning_times == 0:
            for j in range(n):
                if j%thinning_times == 0 and i//thinning_times < new_field.shape[0] and j//thinning_times < new_field.shape[1]:
                    new_field[i//thinning_times, j//thinning_times] = field[i, j]
    return new_field
# Input pre, post, and d and output interpolation differential.
def median_interpolation(front, behind, d):
    return (behind-front)/(2*d)

# double/single version, calculate div_RMS and plot div and savefig
def q1and4_check_averaged_divergence(u, v, title=''): # out 2 lines didn't used, midinter
    if title.split(' ')[-1] == 'double':
        resolution = np.array([2000], dtype=np.float64)[0]
        div = np.zeros((n, n), dtype=np.float64)
        for i in range(2, n-2):
            for j in range(2, n-2):
                div[i][j] = median_interpolation(u[i][j-1], u[i][j+1], resolution) + median_interpolation(v[i-1][j], v[i+1][j], resolution)
        
        div_RMS = 0.0
        for i in range(2, n-2):
            for j in range(2, n-2):
                div_RMS += div[i][j]**2
        div_RMS = (div_RMS/(n-4)**2)**0.5
        print(title+' divergence_RMS', div_RMS)
    elif title.split(' ')[-1] == 'single':
        resolution = np.array([2000], dtype=np.float32)[0]
        div = np.zeros((n, n), dtype=np.float32)
        for i in range(2, n-2):
            for j in range(2, n-2):
                div[i][j] = median_interpolation(u[i][j-1], u[i][j+1], resolution) + median_interpolation(v[i-1][j], v[i+1][j], resolution)
        
        div_RMS = np.array([0.0], dtype=np.float32)[0]
        for i in range(2, n-2):
            for j in range(2, n-2):
                div_RMS += div[i][j]**2
        div_RMS = (div_RMS/(n-4)**2)**0.5
        print(title+' divergence_RMS', div_RMS)
    else:
        print('---q1and4_check_averaged_divergence problem---')
        return

    plt.contourf(div)
    plt.colorbar()
    plt.title(title+', div_RMS='+str(div_RMS))
    plt.savefig('5-'+title+'_div')
    #plt.show()
    plt.close()

# plot wind quiver
def q2and5_plot_wind_quiver(u, v, thinning_times=4, title=''): # thinning and plot
    thinning_u, thinning_v = thinning(u, thinning_times), thinning(v, thinning_times)
    x = np.arange(0, 201-thinning_times, thinning_times)
    y = np.arange(0, 201-thinning_times, thinning_times)
    X, Y = np.meshgrid(x, y)
    plt.quiver(X, Y, thinning_u, thinning_v, scale=5, units='xy', width=0.2) # , headwidth=1, headlength=2
    plt.title(title)
    plt.savefig('5-'+title)
    #plt.show()
    plt.close()

def q3_correct_variationally_adjust(u, v, sd='double'): #強約束風場變分調整，使風場變得完全不可壓縮。上下左右各兩點
# TODO: threshold/mae/mse? eps<1 or >2?
    if sd == 'double': # 雙精度調整:更新完的lamb直接進到下個lamb的計算, eps=1.5 + threshold
        resolution = np.array([2000], dtype=np.float64)[0]
        # check 64bits
        assert type(u[0][0]) == np.float64
        assert type(resolution) == np.float64

        # if lamb has been cal
        filepath = './correct_double_lamb.npy'
        if os.path.isfile(filepath):
            lamb = np.load('correct_double_lamb.npy')
        else:
            # lamb, div init
            lamb = np.zeros((n, n), dtype=np.float64)
            div = np.zeros((n, n), dtype=np.float64)
            for i in range(2, n-2):
                for j in range(2, n-2):
                    div[i][j] = median_interpolation(u[i][j-1], u[i][j+1], resolution) + median_interpolation(v[i-1][j], v[i+1][j], resolution)
            
            # iterate, until every lamb - old_lamb < threshold
            count = 0
            eps = np.array([1.5], dtype=np.float64)[0]
            threshold = np.array([10**-3], dtype=np.float64)[0]
            while True:
                old_lamb = np.array(lamb)
                count += 1

                for i in range(2, n-2):
                    for j in range(2, n-2):
                        F = 2*(resolution**2)*div[i][j] + (lamb[i+2][j]+lamb[i-2][j]+lamb[i][j+2]+lamb[i][j-2])/4
                        lamb[i][j] = lamb[i][j] + eps*(F-lamb[i][j])
                
                skip_flag = False
                max_rela_e = np.array([0.0], dtype=np.float64)[0]
                for i in range(2, n-2):
                    for j in range(2, n-2):
                        if old_lamb[i][j] == np.array([0.0], dtype=np.float64)[0]:
                            max_rela_e = 10*threshold # abs fail
                            skip_flag = True
                        if skip_flag == True:
                            break
                        max_rela_e = max(max_rela_e, abs((lamb[i][j]-old_lamb[i][j])/old_lamb[i][j]))
                    if skip_flag == True:
                        break

                print(count, skip_flag, max_rela_e)

                if max_rela_e < threshold:
                    break
            np.save('correct_double_lamb', lamb)
            print('correct double count:', count)

        #plt.contourf(lamb)
        #plt.colorbar()
        #plt.title('lamb')
        #plt.show()

        # adjust
        for i in range(1, n-1):
            for j in range(1, n-1):
                u[i][j] += 0.5*((lamb[i][j+1]-lamb[i][j-1])/(2*resolution))
                v[i][j] += 0.5*((lamb[i+1][j]-lamb[i-1][j])/(2*resolution))

    else:# 單精度調整 # dtype=np.float32
        resolution = np.array([2000], dtype=np.float32)[0]
        # convert and check 32bits
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        assert type(u[0][0]) == np.float32
        assert type(resolution) == np.float32

        # if lamb has been cal
        filepath = './correct_single_lamb.npy'
        if os.path.isfile(filepath):
            lamb = np.load('correct_single_lamb.npy')
        else:
            # lamb, div init
            lamb = np.zeros((n, n), dtype=np.float32)
            div = np.zeros((n, n), dtype=np.float32)
            for i in range(2, n-2):
                for j in range(2, n-2):
                    div[i][j] = median_interpolation(u[i][j-1], u[i][j+1], resolution) + median_interpolation(v[i-1][j], v[i+1][j], resolution)
            
            # iterate, until every lamb - old_lamb < threshold
            count = 0
            eps = np.array([1.5], dtype=np.float32)[0]
            threshold = np.array([10**-3], dtype=np.float32)[0]
            while True:
                old_lamb = np.array(lamb)
                count += 1

                for i in range(2, n-2):
                    for j in range(2, n-2):
                        F = 2*(resolution**2)*div[i][j] + (lamb[i+2][j]+lamb[i-2][j]+lamb[i][j+2]+lamb[i][j-2])/4
                        lamb[i][j] = lamb[i][j] + eps*(F-lamb[i][j])
                
                skip_flag = False
                max_rela_e = np.array([0.0], dtype=np.float32)[0]
                for i in range(2, n-2):
                    for j in range(2, n-2):
                        if old_lamb[i][j] == np.array([0.0], dtype=np.float32)[0]:
                            max_rela_e = 10*threshold # abs fail
                            skip_flag = True
                        if skip_flag == True:
                            break
                        max_rela_e = max(max_rela_e, abs((lamb[i][j]-old_lamb[i][j])/old_lamb[i][j]))
                    if skip_flag == True:
                        break

                print(count, skip_flag, max_rela_e)

                if max_rela_e < threshold:
                    break
            np.save('correct_single_lamb', lamb)
            print('correct double count:', count)

        #plt.contourf(lamb)
        #plt.colorbar()
        #plt.title('lamb')
        #plt.show()

        # adjust
        for i in range(1, n-1):
            for j in range(1, n-1):
                u[i][j] += 0.5*((lamb[i][j+1]-lamb[i][j-1])/(2*resolution))
                v[i][j] += 0.5*((lamb[i+1][j]-lamb[i-1][j])/(2*resolution))

    return u, v

def q6_wrong_variationally_adjust(u, v, sd='double'): #強約束風場變分調整，使風場變得完全不可壓縮。上下左右各一點
    if sd == 'double': # 雙精度調整:更新完的lamb直接進到下個lamb的計算, eps=1.5 + threshold
        resolution = np.array([2000], dtype=np.float64)[0]
        # check 64bits
        assert type(u[0][0]) == np.float64
        assert type(resolution) == np.float64

        # if lamb has been cal
        filepath = './wrong_double_lamb.npy'
        if os.path.isfile(filepath):
            lamb = np.load('wrong_double_lamb.npy')
        else:
            # lamb, div init
            lamb = np.zeros((n, n), dtype=np.float64)
            div = np.zeros((n, n), dtype=np.float64)
            for i in range(2, n-2):
                for j in range(2, n-2):
                    div[i][j] = median_interpolation(u[i][j-1], u[i][j+1], resolution) + median_interpolation(v[i-1][j], v[i+1][j], resolution)
            
            # iterate, until every lamb - old_lamb < threshold
            count = 0
            eps = np.array([1.5], dtype=np.float64)[0]
            threshold = np.array([10**-3], dtype=np.float64)[0]
            while True:
                old_lamb = np.array(lamb)
                count += 1

                for i in range(2, n-2):
                    for j in range(2, n-2):
                        F = (resolution**2)*div[i][j]/2 + (lamb[i+1][j]+lamb[i-1][j]+lamb[i][j+1]+lamb[i][j-1])/4
                        lamb[i][j] = lamb[i][j] + eps*(F-lamb[i][j])
                
                skip_flag = False
                max_rela_e = np.array([0.0], dtype=np.float64)[0]
                for i in range(2, n-2):
                    for j in range(2, n-2):
                        if old_lamb[i][j] == np.array([0.0], dtype=np.float64)[0]:
                            max_rela_e = 10*threshold # abs fail
                            skip_flag = True
                        if skip_flag == True:
                            break
                        max_rela_e = max(max_rela_e, abs((lamb[i][j]-old_lamb[i][j])/old_lamb[i][j]))
                    if skip_flag == True:
                        break

                print(count, skip_flag, max_rela_e)

                if max_rela_e < threshold:
                    break
            np.save('wrong_double_lamb', lamb)
            print('wrong double count:', count)

        #plt.contourf(lamb)
        #plt.colorbar()
        #plt.title('lamb')
        #plt.show()

        # adjust
        for i in range(1, n-1):
            for j in range(1, n-1):
                u[i][j] += 0.5*((lamb[i][j+1]-lamb[i][j-1])/(2*resolution))
                v[i][j] += 0.5*((lamb[i+1][j]-lamb[i-1][j])/(2*resolution))

    else:# 單精度調整 # dtype=np.float32
        resolution = np.array([2000], dtype=np.float32)[0]
        # convert and check 32bits
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        assert type(u[0][0]) == np.float32
        assert type(resolution) == np.float32

        # if lamb has been cal
        filepath = './wrong_single_lamb.npy'
        if os.path.isfile(filepath):
            lamb = np.load('wrong_single_lamb.npy')
        else:
            # lamb, div init
            lamb = np.zeros((n, n), dtype=np.float32)
            div = np.zeros((n, n), dtype=np.float32)
            for i in range(2, n-2):
                for j in range(2, n-2):
                    div[i][j] = median_interpolation(u[i][j-1], u[i][j+1], resolution) + median_interpolation(v[i-1][j], v[i+1][j], resolution)
            
            # iterate, until every lamb - old_lamb < threshold
            count = 0
            eps = np.array([1.5], dtype=np.float32)[0]
            threshold = np.array([10**-3], dtype=np.float32)[0]
            while True:
                old_lamb = np.array(lamb)
                count += 1

                for i in range(2, n-2):
                    for j in range(2, n-2):
                        F = (resolution**2)*div[i][j]/2 + (lamb[i+1][j]+lamb[i-1][j]+lamb[i][j+1]+lamb[i][j-1])/4
                        lamb[i][j] = lamb[i][j] + eps*(F-lamb[i][j])
                
                skip_flag = False
                max_rela_e = np.array([0.0], dtype=np.float32)[0]
                for i in range(2, n-2):
                    for j in range(2, n-2):
                        if old_lamb[i][j] == np.array([0.0], dtype=np.float32)[0]:
                            max_rela_e = 10*threshold # abs fail
                            skip_flag = True
                        if skip_flag == True:
                            break
                        max_rela_e = max(max_rela_e, abs((lamb[i][j]-old_lamb[i][j])/old_lamb[i][j]))
                    if skip_flag == True:
                        break

                print(count, skip_flag, max_rela_e)

                if max_rela_e < threshold:
                    break
            np.save('wrong_single_lamb', lamb)
            print('wrong double count:', count)

        #plt.contourf(lamb)
        #plt.colorbar()
        #plt.title('lamb')
        #plt.show()

        # adjust
        for i in range(1, n-1):
            for j in range(1, n-1):
                u[i][j] += 0.5*((lamb[i][j+1]-lamb[i][j-1])/(2*resolution))
                v[i][j] += 0.5*((lamb[i+1][j]-lamb[i-1][j])/(2*resolution))

    return u, v


#### test
def q3_cva_test(u, v, sd='double'): # mse test
    if sd == 'double': # 雙精度調整:更新完的lamb直接進到下個lamb的計算, eps=1.5 + mse
        resolution = np.array([2000], dtype=np.float64)[0]
        # check 64bits
        assert type(u[0][0]) == np.float64
        assert type(resolution) == np.float64

        # if lamb has been cal
        filepath = './test_correct_double_lamb.npy'
        if os.path.isfile(filepath):
            lamb = np.load('test_correct_double_lamb.npy')
        else:
            # lamb, div init
            lamb = np.zeros((n, n), dtype=np.float64)
            div = np.zeros((n, n), dtype=np.float64)
            for i in range(2, n-2):
                for j in range(2, n-2):
                    div[i][j] = median_interpolation(u[i][j-1], u[i][j+1], resolution) + median_interpolation(v[i-1][j], v[i+1][j], resolution)
            
            # iterate, until every lamb - old_lamb < threshold
            count = 0
            eps = np.array([1.5], dtype=np.float64)[0]
            mse_threshold = np.array([10**-2], dtype=np.float64)[0]
            while True:
                old_lamb = np.array(lamb)
                count += 1

                for i in range(2, n-2):
                    for j in range(2, n-2):
                        F = 2*(resolution**2)*div[i][j] + (lamb[i+2][j]+lamb[i-2][j]+lamb[i][j+2]+lamb[i][j-2])/4
                        lamb[i][j] = lamb[i][j] + eps*(F-lamb[i][j])
                
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

                print(count, skip_flag, mse)

                if mse < mse_threshold:
                    break
                #if count > 30:
                #    break
            np.save('test_correct_double_lamb', lamb)
            print('test correct double count:', count)

        #plt.contourf(lamb)
        #plt.colorbar()
        #plt.title('lamb')
        #plt.show()

        # adjust
        for i in range(1, n-1):
            for j in range(1, n-1):
                u[i][j] += 0.5*((lamb[i][j+1]-lamb[i][j-1])/(2*resolution))
                v[i][j] += 0.5*((lamb[i+1][j]-lamb[i-1][j])/(2*resolution))
    else:
        print(':((((')
    return u, v

if __name__ == '__main__':
    u, v = read_data()
    q1and4_check_averaged_divergence(u, v, 'origin double')
    q2and5_plot_wind_quiver(u, v, 4, 'origin double')

    cva_s_u, cva_s_v = q3_correct_variationally_adjust(u, v, 'single')
    q1and4_check_averaged_divergence(cva_s_u, cva_s_v, 'correct single')
    q2and5_plot_wind_quiver(cva_s_u, cva_s_v, 4, 'correct single')

    cva_d_u, cva_d_v = q3_correct_variationally_adjust(u, v, 'double')
    q1and4_check_averaged_divergence(cva_d_u, cva_d_v, 'correct double')
    q2and5_plot_wind_quiver(cva_d_u, cva_d_v, 4, 'correct double')

    wva_s_u, wva_s_v = q6_wrong_variationally_adjust(u, v, 'single')
    q1and4_check_averaged_divergence(wva_s_u, wva_s_v, 'wrong single')
    q2and5_plot_wind_quiver(wva_s_u, wva_s_v, 4, 'wrong single')

    wva_d_u, wva_d_v = q6_wrong_variationally_adjust(u, v, 'double')
    q1and4_check_averaged_divergence(wva_d_u, wva_d_v, 'wrong double')
    q2and5_plot_wind_quiver(wva_d_u, wva_d_v, 4, 'wrong double')



    cva_d_u_test, cva_d_v_test = q3_cva_test(u, v, 'double')
    q1and4_check_averaged_divergence(cva_d_u_test, cva_d_v_test, 'correct test double')
    q2and5_plot_wind_quiver(cva_d_u_test, cva_d_v_test, 4, 'correct test double')