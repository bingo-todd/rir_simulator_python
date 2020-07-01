import numpy as np
EPSILON = 1e-20


def A2RT(room_size, A_wall_all, F_abs, c=343, A_air=None, estimator='Norris_Eyring'):
    """ Estimate reverberation time based on room acoustic parameters,
    translated from matlab code developed by Douglas R Campbell
    Args:
        room_size: three-dimension measurement of shoebox room
        A_wall_all: sound absorption coefficients of six wall surfaces
        c: sound speed, default to 343 m/s
        F_abs: center frequency of each frequency band
        A_air: absorption coefficients of air, if not specified, it will
            calculated based on humidity of 50
        estimator: estimate methods, choose from [Sabine,SabineAir,
            SabineAirHiAbs,Norris_Eyring], default to Norris_Eyring
    """

    if A_air is None:
        humidity = 50
        A_air = (5.5e-4)*(50/humidity)*((F_abs/1000)**1.7)

    Lx, Ly, Lz = room_size
    V_room = np.prod(room_size)  # Volume of room m^3
    S_wall_all = [Lx*Lz, Ly*Lz, Lx*Ly]
    S_room = 2.*np.sum(S_wall_all)  # Total area of shoebox room surfaces
    # Effective absorbing area of room surfaces at each frequency
    Se = (S_wall_all[1]*(A_wall_all[:, 0] + A_wall_all[:, 1])
          + S_wall_all[0]*(A_wall_all[:, 2] + A_wall_all[:, 3])
          + S_wall_all[2]*(A_wall_all[:, 4] + A_wall_all[:, 5]))
    A_mean = Se/S_room  # Mean absorption of wall surfaces
    # Mean absorption of air averaged across frequency.
    # A_air_mean = np.mean(A_air)
    # Mean Free Path (Average distance between succesive reflections) (Ref A4)
    # MFP = 4*V_room/S_room

    # Reverberation time estimate
    # Detect anechoic case and force RT60 all zeros
    if np.linalg.norm(1-A_mean) < EPSILON:
        RT60 = np.zeros(F_abs.shape)
    else:  # Select an estimation equation
        if estimator == 'Sabine':
            RT60 = np.divide((55.25/c)*V_room, Se)  # Sabine equation
        if estimator == 'SabineAir':
            # Sabine equation (SI units) adjusted for air
            RT60 = np.divide((55.25/c)*V_room, (4*A_air*V_room+Se))
        if estimator == 'SabineAirHiAbs':
            # % Sabine equation (SI units) adjusted for air and high absorption
            RT60 = np.divide(55.25/c*V_room,
                             4*A_air*V_room+np.multiply(Se, (1+A_mean/2)))
        if estimator == 'Norris_Eyring':
            # Norris-Eyring estimate adjusted for air absorption
            RT60 = np.divide(55.25/c*V_room,
                             4*A_air*V_room-S_room*np.log(1-A_mean+EPSILON))

        return RT60


def RT2A(RT60, F_abs, room_size, c=343, A_air=None,
         estimator='Norris_Eyring'):
    Lx, Ly, Lz = room_size
    V_room = np.prod(room_size)  # Volume of room m^3
    S_wall_all = [Lx*Lz, Ly*Lz, Lx*Ly]
    S_room = 2.*np.sum(S_wall_all)  # Total area of shoebox room surfaces

    if A_air is None:
        humidity = 50
        A_air = (5.5e-4)*(50/humidity)*((F_abs/1000)**1.7)

    if estimator == 'Sabine':
        coef = np.divide(55.25/c*V_room/S_room, RT60)
    if estimator == 'SabineAir':
        coef = (np.divide(55.25/c*V_room, RT60)-4*A_air*V_room)/S_room
    if estimator == 'SabineAirHiAbs':
        coef = np.sqrt(
                2*(np.divide(55.25/c*V_room, RT60) - 4*A_air*V_room) + 1) - 1
    if estimator == 'Norris_Eyring':
        coef = 1-np.exp((4*A_air*V_room
                         - np.divide(55.25/c*V_room, RT60)
                         )/S_room)

    return coef


def test_A2RT():
    F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])
    # A_acoustic_plaster = np.asarray([0.10,0.20,0.50,0.60,0.70,0.70])
    # A_RT0_5 = np.asarray([0.2136,0.2135,0.2132,0.2123,0.2094,0.1999])
    A_RT0_5 = np.asarray([0.2214, 0.3051, 0.6030, 0.7770, 0.8643, 0.8627])
    room_size = (5.1, 7.1, 3)
    A_wall_all = np.repeat(A_RT0_5.reshape((-1, 1)), repeats=6, axis=1)
    RT = A2RT(room_size=room_size, A_wall_all=A_wall_all, F_abs=F_abs)
    print(RT)


def test_RT2A():
    F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])
    A_RT0_5 = np.asarray([0.2136, 0.2135, 0.2132, 0.2123, 0.2094, 0.1999])
    room_size = (5.1, 7.1, 3)
    A_wall_all = np.repeat(A_RT0_5.reshape((-1, 1)), repeats=6, axis=1)
    RT = A2RT(room_size=room_size, A_wall_all=A_wall_all, F_abs=F_abs)
    coef = RT2A(RT60=RT, room_size=room_size, F_abs=F_abs)
    print(coef)


if __name__ == '__main__':
    test_A2RT()
