import torch


def euler(fun, t_start, t_end, y0, time_step):
    """
    ODE solver implementing method Euler (RK1)

    Parameters:
    fun: callable
        Right-hand side of the system, the calling signature is fun(t, y)
    t_start, t_end: double
        Time interval of integration. The solver starts with t=t_start and integrates until it reaches t=t_end.
        If t_start > t_end, ODE will be solved backwards.
    y0: double, array-like
        Initial values of the ivp
    time_step: double
        step size of RK4, positive
    :return:
    y: ndarray, shape (n_steps, )
        array of values in time order.
    """
    if t_start > t_end:
        time_step = -time_step
    n_steps = round((t_end - t_start) / time_step)
    y_matrix = [y0]
    t = t_start
    for _ in range(n_steps):
        yt = y0 + time_step * fun(t, y0)
        # yt = torch.clamp(yt, min=0)
        y_matrix.append(yt)
        y0 = yt
        t = t + time_step
    if t_start > t_end:
        y_matrix.reverse()
    return torch.stack(y_matrix, dim=0)


def rk4(fun, t_start, t_end, y0, time_step):
    """
    ODE solver implementing method RK4

    Parameters:
    fun: callable
        Right-hand side of the system, the calling signature is fun(t, y)
    t_start, t_end: double
        Time interval of integration. The solver starts with t=t_start and integrates until it reaches t=t_end.
        If t_start > t_end, ODE will be solved backwards.
    y0: double, array-like
        Initial values of the ivp
    time_step: double
        step size of RK4, positive
    :return:
    y: ndarray, shape (n_steps, )
        array of values in time order.
    """
    if t_start > t_end:
        time_step = -time_step
    n_steps = round((t_end - t_start) / time_step)
    y_matrix = [y0]
    t = t_start
    for _ in range(n_steps):
        k1 = fun(t, y0)
        k2 = fun(t + time_step / 2, y0 + time_step / 2 * k1)
        k3 = fun(t + time_step / 2, y0 + time_step / 2 * k2)
        k4 = fun(t + time_step, y0 + time_step * k3)
        yt = y0 + 1 / 6 * time_step * (k1 + 2 * k2 + 2 * k3 + k4)
        y_matrix.append(yt)
        y0 = yt
        t += time_step
    if t_start > t_end:
        y_matrix.reverse()
    return torch.stack(y_matrix, dim=0)


def rk14(fun, t_start, t_end, y0, time_step):
    """
    ODE solver implementing method RK4

    Parameters:
    fun: callable
        Right-hand side of the system, the calling signature is fun(t, y)
    t_start, t_end: double
        Time interval of integration. The solver starts with t=t_start and integrates until it reaches t=t_end.
        If t_start > t_end, ODE will be solved backwards.
    y0: double, array-like
        Initial values of the ivp
    time_step: double
        step size of RK4, positive
    :return:
    y: ndarray, shape (n_steps, )
        array of values in time order.
    """
    pass


def rk45(fun, t_start, t_end, y0, time_step):
    """
    ODE solver implementing method RK4

    Parameters:
    fun: callable
        Right-hand side of the system, the calling signature is fun(t, y)
    t_start, t_end: double
        Time interval of integration. The solver starts with t=t_start and integrates until it reaches t=t_end.
        If t_start > t_end, ODE will be solved backwards.
    y0: double, array-like
        Initial values of the ivp
    time_step: double
        step size of RK4, positive
    :return:
    y: ndarray, shape (n_steps, )
        array of values in time order.
    """
    pass
