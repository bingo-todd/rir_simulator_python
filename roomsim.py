import numpy as np
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
