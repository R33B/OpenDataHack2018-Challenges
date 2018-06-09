from datetime import datetime, timedelta


def days_of_month(y, m):
    d0 = datetime(y, m, 1)
    d1 = datetime(y, m + 1, 1)
    out = list()
    while d0 < d1:
        out.append(d0.strftime('%Y-%m-%d'))
        d0 += timedelta(days=1)
    return out
