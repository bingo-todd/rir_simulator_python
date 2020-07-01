import numpy as np
<<<<<<< HEAD
from RT60 import RT2A

F_abs = [125, 250, 500, 1000, 2000, 4000, 8000]


class Room(object):
    def __init__(self, room_size, RT=None, A=None):
        self.room_size = room_size
        if RT is None:
            A = RT2A(RT, room_size)


class RoomSim:
    def __init__(self):
        self.Fs = 44100
        self.src_pos = np.array(src_pos)
        self.RT60 = np.ones(6) 
        return 

    def hi_pass2(self, Fc, Fs):
        T = 1/Fs
        w = 2*np.pi*Fc
        r1 = np.exp(-w*T)
        r2 = r1
        a1, a2 = -(1+r2), r2
        b1, b2 = 2*r1*np.cos(w*T), -r1**2
        gain = (1-b1+b2)/(1+a1-a2)
        b = np.asarray([1, b1, b2]/gain)
        a = [1, -a1, -a2]
        return b, a

    def cal_dist(pos1, pos2):
        """pos1, pos2: [x, y, z]
        """
        return np.sqrt(np.sum((pos1-pos2)**2))

    def init_fft_config(self):
        self.F_abs_norm = self.F_abs/F_nyquist 
        self.N_fft = 512 
        self.N_fft_half = 256
        self.window = np.hanning(N_refl+1)
       
    def rir_per_image(self, b_refl):     
        # Estimate the values of reflection coefficient at the linear
        # interpolated grid points
        interp_func = interp1d(self.F_abs_norm, b_refl)
        b_refl = interp_func(1.0/Half_I*np.arange(Half_I+1))
        
        b_refl = np.hstack((b_refl, b_refl[::-1][1:-1]))
        h_refl = np.real(
                    np.fft.ifft(np.concatenate((b_refl, b_refl[1:-2], N_refl))
        # Make the impulse realisable (half length shift) and Hann window it
        h_refl = window*np.hstack((h_refl[Half_I_plusone-1:N_refl], h_refl[:Half_I_plusone]))
        return h_refl
 

    def syn_rir(self, src_pos, source_off=None, source_dir=None):
        '''Create the RIR
        src_pos : list containing xyz position of the source
        source_off: 3 x 1 list representing the source orientation (azimuth,
        elevation, roll)
        source_dir: source directivity np txt file of dimension 181 x 361
        '''
        # constants
        Two_pi = 2*np.pi
        sound_speed = 343.0

        T_Fs = 1.0/self.Fs
        F_nyquist = self.Fs/2.0 # Half sampling frequency
        Fs_c = self.Fs/sound_speed # Samples per metre

        # Reflection order and impulse response length
        H_length = np.floor(np.max(self.RT60)*self.Fs)
         
        order_refl_all = np.ceil(H_length/Fs_C/(2*self.room_size))
        order_refl_all = np.ones(3)*10

        # Maximum number of image sources
        n_img_src = np.prod(2*order_refl_all+1)
        delay_s = Fs_c*self.cal_dist(src_pos, self.mic_pos)

        #TODO not in matlab 
        H_length = np.int32(
                        np.max((H_length, 
                                np.ceil(np.max(delay_s))+200)))
       
        # Smooth filter
        # Interpolation filter for fractional delays
        N_frac = 32 # Order of FIR fractional delay filter
        Fc = 0.9*F_nyquist
        Fc_Fs = Fc*T
        Two_Fc = 2*Fc
        T_win = N_frac*T_Fs # Window duration (seconds)
        Two_pi_Tw = Two_pi/T_win # Compute here for efficiency
        t = np.arange(-T_win/2, T_win/2+T_Fs, T_Fs)
        pad_frac = np.zeros((N_frac,1))
        
        # Second order high-pass IIR filter to remove DC buildup (nominal -4dB cut-off at 20 Hz)
        if True:
            b_HP, a_HP = self.hi_pass2(Fc, Fs)

        # Further constants
        room_size_double = 2*self.room_size

        #codes the eight permutations of x+/-xp, y+/-yp, z+/-zp
        #(the source to receiver vector components) where [-1 -1 -1] identifies the parent source.
        img_src_ident = np.array([[-1, -1, -1],
                                  [-1, -1, +1],
                                  [-1, +1, -1],
                                  [-1, +1, +1],
                                  [+1, -1, -1],
                                  [+1, -1, +1],
                                  [+1, +1, -1],
                                  [+1, +1, +1]])
        # Includes/excludes bx, by, bz depending on 0/1 state.
        surface_coeff = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 1],
                                  [1, 0, 0],
                                  [1, 0, 1],
                                  [1, 1, 0],
                                  [1, 1, 1]])
        # 
        qq = surface_coeff[:,0] #  for B[0]
        jj = surface_coeff[:,1] #  for B[2]
        kk = surface_coeff[:,2] #  for B[4]
       
        # TODO Problem 
        #Image locations and impulse responses
        pos_img_src = np.zeros((n_img_src, 3)) # image source co-ordinates
        refl = np.zeros((n_img_src, len(self.F_abs)) # surface reflection impulse amplitude 
        pos_relative_all = img_src_ident * src_pos[np.newaxis, :]
        n_image=-1; #number of significant images of each parent source
        B = np.sqrt(1-self.A);
        B[0], B[1], B[2], B[3], B[4], B[5] = B

        for i_x in np.arange(-order_x, order_x+1):
            atten_x2 = B[1]**np.abs(i_x)
            pos_x = i_x*room_size_double[0]
            for i_y in np.arange(-order_y, order_y):
                atten_x2y2 = atten_x2*(B[3]**np.abs(i_y))
                pos_y = i_y*self.room_size_double[1]
                for i_z in np.arange(-order_z, order_z+1):
                    atten_x2y2z2 = atten_x2y2*(B[5]**np.abs(i_z))
                    pos_z = i_z*room_size_double[2]
                    pos_tmp = np.asarray([pos_x, pos_y, pos_z])
                    for permu in np.arange(8):
                        n_image = n_image+1
                        pos_img_src[:,n_image] =  pos_tmp - pos_relative_all[permu] 
                        delay = Fs_c*self.cal_dist(pos_img_src[:, n_image], self.mic_pos)
                        # compute only for image sources within impulse response length
                        if delay <= H_length:
                            atten_x1y1z1 = ((B[0]**np.abs(i_x-qq[permu]))
                                            * (B[2]**np.abs(i_y-jj[permu]))
                                            * (B[4]**np.abs(i_z-kk[permu])))
                            refl[n_image:] = atten_x1y1z1*atten_x2y2z2 
                            if np.sum(refl[n_image:]) < 1E-6:
                                n_image=n_image-1
                        else:
                            # Delete image sources with a delay > length H_length
                            n_image=n_image-1

        return refl, pos_img_src
=======
from scipy.interpolate import interp1d
import scipy.signal as dsp


def RT2Absorb(RT60, F_abs, room_size, c=343, A_air=None,
              estimator='Norris_Eyring'):
    V_room = np.prod(room_size)  # Volume of room m^3
    S_room = 2.*(room_size[0]*room_size[1]
                 + room_size[0]*room_size[2]
                 + room_size[1]*room_size[2])

    if A_air is None:
        humidity = 50
        A_air = (5.5e-4)*(50/humidity)*((F_abs/1000)**1.7)

    if estimator == 'Sabine':
        A = np.divide(55.25/c*V_room/S_room, RT60)
    if estimator == 'SabineAir':
        A = (np.divide(55.25/c*V_room, RT60)-4*A_air*V_room)/S_room
    if estimator == 'SabineAirHiAbs':
        A = np.sqrt(
                2*(np.divide(55.25/c*V_room, RT60) - 4*A_air*V_room) + 1) - 1
    if estimator == 'Norris_Eyring':
        A = 1-np.exp((4*A_air*V_room
                      - np.divide(55.25/c*V_room, RT60))/S_room)

    return A


class RoomSim:

    def __init__(self):
        self.room_size = np.asarray([10, 10, 10])
        self.RT60 = 0.6
        self.F_abs = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
        self.Fs = 16000
        self.T_Fs = 1./self.Fs

        self.A = RT2Absorb(self.RT60, self.F_abs, self.room_size)
        self.B = np.sqrt(1 - self.A)

        # constants
        self.Two_pi = 2*np.pi
        self.c_air = 343.0  # sound speed in air
        self.T_Fs = 1.0/self.Fs
        self.F_nyquist = self.Fs/2.0  # Half sampling frequency
        self.Fs_c = self.Fs/self.c_air  # Samples per metre

        # Interpolation filter for fractional delays
        N_frac = 32  # Order of FIR fractional delay filter
        T_window = N_frac*self.T_Fs  # Window duration (seconds)
        self.Two_pi_Tw = self.Two_pi/T_window  # Compute here for efficiency
        self.t = np.arange(-T_window/2, T_window/2+self.T_Fs, self.T_Fs)
        self.pad_frac = np.zeros((N_frac, 1))

        #
        b_HP, a_HP = self.make_HP_filter()

        # generate single reflection rir based on spectrum amplitude
        self.F_abs_N = self.F_abs/self.F_nyquist
        self.n_fft = int(2*np.round(self.F_nyquist/self.F_abs[1]))
        self.n_fft_half = int(self.n_fft/2.)
        self.window = np.hanning(self.n_fft)

        self.m_air = 6.875e-4*(self.F_abs/1000)**(1.7)

        self.rir_length = np.floor(np.max(self.RT60)*self.Fs)
        self.refl_order = np.ceil(self.rir_length/self.Fs_c/(self.room_size*2))

        # init
        self.refl_order = np.zeros(3)
        self.refl_amp = np.zeros(0, self.F_abs.shape[0])
        self.n_img = 0

    def make_HP_filter(self):
        # Second order high-pass IIR filter to remove DC buildup
        # (nominal -4dB cut-off at 20 Hz)
        w = 2*np.pi*20
        r1, r2 = np.exp(-w*self.T_Fs), np.exp(-w*self.T_Fs)
        b1, b2 = -(1+r2), np.copy(r2)  # Numerator coefficients (fix zeros)
        a1, a2 = 2*r1*np.cos(w*self.T_Fs), -r1*r1  # Denominator coefficients
        HP_gain = (1-b1+b2)/(1+a1-a2)  # Normalisation gain
        b = [1, b1, b2]/HP_gain
        a = [1, -a1, -a2]
        return b, a

    def cal_b_power(self, i, n):
        """i: index of B
        n: power term
        """
        key = f'(i)_{n}'
        if key not in self.B_power_table.keys():
            self.B_power_table[key] = self.B[i]**n
        return self.B_power_table[key]

    def cal_B_power(self, n_all):
        result = np.ones(self.F_abs.shape[0])
        for wall_i in range(6):
            result = result * self.cal_b_power(wall_i, n_all[wall_i])
        return result

    def cal_dist(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def get_all_refl(self, src_pos=[2, 3, 4], source_off=None,
                     source_dir=None):
        # TODO input argments validate

        src_pos = np.asarray(src_pos)

        # Constant
        self.B_power_table = dict()
        # not used in BRIR
        if False:
            if source_dir is None:  # omnidirectional
                source_dir = np.ones((181, 361))
            else:
                source_dir = np.loadtxt(source_dir)
            if source_off is None:
                source_off = np.zeros(src_pos.shape)

        # Maximum number of image sources
        n_img = np.int64(np.prod(2*self.refl_order+1)*8)

        # codes the eight permutations of x+/-xp, y+/-yp, z+/-zp
        # where [-1 -1 -1] identifies the parent source.
        img_ident_in_cube = np.array([[+1, +1, +1],
                                      [+1, +1, -1],
                                      [+1, -1, +1],
                                      [+1, -1, -1],
                                      [-1, +1, +1],
                                      [-1, +1, -1],
                                      [-1, -1, +1],
                                      [-1, -1, -1]])
        img_pos_in_cube = img_ident_in_cube*src_pos[np.newaxis, :]
        # Includes/excludes bx, by, bz depending on 0/1 state.
        refl_order_relative = np.array([[+0, +0, +0],
                                        [+0, +0, -1],
                                        [+0, -1, +0],
                                        [+0, -1, -1],
                                        [-1, +0, +0],
                                        [-1, +0, -1],
                                        [-1, -1, +0],
                                        [-1, -1, -1]])

        # Image locations and impulse responses
        img_pos = np.zeros((n_img, 3))  # image source co-ordinates
        n_F_abs = len(self.F_abs)
        refl_amp = np.zeros((n_img, n_F_abs))
        n_img = -1  # number of significant images of each parent source

        room_size_double = 2*self.room_size
        # i_x1, i_x2, i_y1, i_y2, i_z1, i_z2, reflection number of each wall
        refl_order = self.refl_order
        for i_x2 in np.arange(-refl_order[0], refl_order[0]+1):
            disp_x = i_x2*room_size_double[0]
            for i_y2 in np.arange(-refl_order[1], refl_order[1]+1):
                disp_y = i_y2*room_size_double[1]
                for i_z2 in np.arange(-refl_order[2], refl_order[2]+1):
                    disp_z = i_z2*room_size_double[2]
                    pos_cube = [disp_x, disp_y, disp_z]
                    for permu in np.arange(8):
                        n_img = n_img+1
                        img_pos[n_img] = pos_cube + img_pos_in_cube[permu]
                        i_x1 = np.abs(i_x2+refl_order_relative[permu, 1])
                        i_y1 = np.abs(i_y2+refl_order_relative[permu, 1])
                        i_z1 = np.abs(i_z2+refl_order_relative[permu, 2])
                        refl_amp[n_img] = self.cal_B_power([i_x1, i_x2,
                                                            i_y1, i_y2,
                                                            i_z1, i_z2])
        # Complete impulse response for the source
        n_img = n_img + 1  # why add 1, the real source
        img_pos = img_pos[:, :n_img]
        refl_amp = refl_amp[:, :n_img]

        self.img_pos = img_pos
        self.refl_amp = refl_amp
        self.n_img

    def cal_rir(self):
        source_off = 0
        [c_psi, s_psi,
         c_theta, s_theta,
         c_phi, s_phi] = self.__create_psi_theta_phi(source_off)

        tm_source = self.__create_tm([c_psi, s_psi, c_theta, s_theta,
                                      c_phi, s_phi])

        rir = np.zeros((self.rir_length, self.n_channel))
        # attenuation factors for one metre travelled in air
        temp_count = 0
        atten_air = np.exp(-0.5*self.m_air).T

        for mic in self.mics:
            # Get the sensor direction-dependent impulse responses
            sensor_dir = mic.direction
            sensor_dir = np.loadtxt(sensor_dir+'.txt')
            sensor_No = int(mic._id)-1
            # for each of the n_img image sources
            for img_i in np.arange(self.n_img):
                amp = self.refl_amp[:, img_i]
                # Position vector from sensor_No to source(img_i)
                disp = self.img_pos[img_i]-self.sensor_xyz[sensor_No, :]
                # Distance (m) between image source(img_i) and sensor_No
                dist = np.sqrt(np.sum(disp**2))

                # Include effect of distance (ie. 1/R) attenuation
                amp = amp/dist

                # Include the absorption due to air
                amp = amp*(atten_air**dist)

                # Estimate the values of reflection coefficient at the linear
                # interpolated grid points
                amp_func = interp1d(self.F_abs_N, amp)
                amp = amp_func(
                    1.0/self.n_fft_half*np.arange(self.n_fft_half+1))

                amp = np.hstack((amp, amp[::-1][1:-1]))
                h_refl = np.real(np.fft.ifft(amp, self.n_fft))
                h_refl = self.window*np.hstack(
                                    (h_refl[self.n_fft_half:self.n_fft],
                                     h_refl[:self.n_fft_half+1]))

                # For primary sources, and image sources with impulse response
                # peak magnitudes >= -100dB (1/100000)
                if ((self.n_img == 1)
                        or np.max(np.abs(h_refl[:self.n_fft_half+1])) >= 1E-5):
                    # Fractional delay filter
                    delay = self.Fs_c*dist
                    rdelay = np.round(delay)
                    t_Td = self.t-(delay-rdelay)*self.T_Fs
                    hsf=.5*(1+np.cos(self.Two_pi_Tw*t_Td))*np.sinc(self.Fs*t_Td); # Compute delayed filter impulse response for sensor
                    # Convolve channel signals
                    sig_to_conv = np.vstack((h_refl.reshape(len(h_refl), 1),
                                             self.pad_frac))
                    sig_to_conv = sig_to_conv.reshape(len(sig_to_conv),)
                    h = dsp.lfilter(hsf, 1, sig_to_conv)
                    len_h=len(h); # length of impulse response modelling image source response
                    adjust_delay = int(rdelay - np.ceil(len_h/2.0)) # Half length shift to remove delay due to impulse response
                    # Sensor filter
                    # position vector from each sensor location to each image source in sensor axes system
                    xyz_source = np.dot(self.tm_sensor[sensor_No, :, :], xyz)
                    # Distance (m) between sensor_No and proj of image source on xy plane
                    hyp = np.sqrt(xyz_source[0]**2+xyz_source[1]**2);
                    elevation = np.arctan(xyz_source[2]/(hyp+np.finfo(float).eps)); # Calculate -pi/2 <= elevation <= +pi/2 rads
                    azimuth = np.arctan2(xyz_source[1],xyz_source[0]); # Calculate -pi <= azimuth <= +pi rad
                    e_index = int(np.round(elevation*180/np.pi)+90)
                    a_index = int(np.round(azimuth*180/np.pi)+180)
                    sensor_ir=[sensor_dir[e_index,a_index]]
                    #h=dsp.lfilter(sensor_ir,1,np.hstack((h, np.zeros((len(sensor_ir)-1,1)))))
                    h = dsp.lfilter(sensor_ir, 1, h)
                    # Source filter
                    # position vector from each image source location to each sensor in source axes system
                    xyz_sensor = -1 * np.dot(tm_source, xyz)
                    # Distance (m) between image source and proj of sensor_No on xy plane
                    hyp = np.sqrt(xyz_sensor[0]**2 + xyz_sensor[1]**2)
                    # Calculate -pi/2 <= elevation <= +pi/2 rads
                    elevation=np.arctan(xyz_sensor[2]/(hyp+np.finfo(float).eps))
                    # Calculate -pi <= azimuth <= +pi rad
                    azimuth=np.arctan2(xyz_sensor[1],xyz_sensor[0])
                    e_index = int(np.round(elevation*180/np.pi)+90)
                    a_index = int(np.round(azimuth*180/np.pi)+180)
                    source_ir = [source_dir[e_index, a_index]]
                    #h = dsp.lfilter(source_ir,1,np.hstack((h, np.zeros((len(source_ir)-1,1)))))
                    h = dsp.lfilter(source_ir, 1, h)
                    len_h = len(h);
                    #Accumulate the impulse responses from each image source within an array of length rir_length
                    start_index_Hp = max(adjust_delay+(adjust_delay >= 0), 0)
                    stop_index_Hp = min(adjust_delay+len_h, self.rir_length)
                    start_index_h = max(-adjust_delay, 0)
                    stop_index_h = start_index_h + (stop_index_Hp - start_index_Hp)
                    #print(temp_count, start_index_Hp, stop_index_Hp, start_index_h, stop_index_h)
                    temp_count += 1
                    if stop_index_h < 0:
                        continue
                    #Add whole or part of impulse response
                    rir[start_index_Hp:stop_index_Hp, sensor_No] = rir[start_index_Hp:stop_index_Hp, sensor_No] + h[start_index_h:stop_index_h]
            #High-pass filtering
            rir[:, sensor_No] = dsp.lfilter(self.b_HP, self.a_HP, rir[:, sensor_No])
        return rir


if __name__ == '__main__':
    roomsim = RoomSim()
    roomsim.get_all_refl()
>>>>>>> bd7d30e1419560668878fc258aa974b3f40f251e
