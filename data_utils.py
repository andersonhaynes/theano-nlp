import urllib

def get_file(fpath, origin):
    try:
        f = open(fpath)
    except:
        print 'Downloading data from',  origin
        urllib.urlretrieve(origin, fpath)
    return fpath