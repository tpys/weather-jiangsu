import argparse
import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import pandas as pd
import pygrib
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--models', type=str, nargs='+', default=[""])
parser.add_argument('--future_frames', type=int, default=12)


args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)


obs_dir = os.path.join(args.data_dir, 'obs')
wrf_dir = os.path.join(args.data_dir, 'wrf')
model_dir = os.path.join(args.data_dir, 'models')
meta_name = os.path.join(args.data_dir, 'meta_info.json')



with open(meta_name) as f:
    meta_info = json.load(f)
    
    obs_mean = np.array(meta_info['obs_mean'], dtype=np.float32)
    obs_std = np.array(meta_info['obs_std'], dtype=np.float32)
    wrf_mean = np.array(meta_info['wrf_mean'], dtype=np.float32)
    wrf_std = np.array(meta_info['wrf_std'], dtype=np.float32)
    
    wrf_names = meta_info['wrf_names']  
    wrf_levels = meta_info['wrf_levels']
    
    out_rect = meta_info['out_rect']    
    wrf_rect = meta_info['wrf_rect']        
    obs_latlon = meta_info['obs_latlon']
    out_latlon = meta_info['out_latlon']
    out_degree = meta_info['out_degree']
    input_size = meta_info['input_size']
    input_frames = meta_info['input_frames']
    
    print("model -> input_frames: {}, input_size: {}".format(input_frames, input_size))
    print("obs -> latlon: {}, mean: {}, std: {}".format(obs_latlon, obs_mean, obs_std))
    print("wrf -> levels: {}, pl: {}, sfc: {}".format(wrf_levels, wrf_names['pl'], wrf_names['sfc']))
    
    obs_mean = obs_mean.reshape(-1, 1, 1)
    obs_std = obs_std.reshape(-1, 1, 1)
    wrf_mean = wrf_mean.reshape(-1, 1, 1)
    wrf_std = wrf_std.reshape(-1, 1, 1)
    
        
    
def convert_time(time, zone='utc2bj', fmt='%Y%m%d%H'):
    # time = pd.to_datetime(str(time), format=fmt)

    if zone == 'utc2bj':
        time = time + pd.Timedelta(hours=8)
    if zone == 'bj2utc':
        time = time - pd.Timedelta(hours=8)

    return time
# .strftime(fmt)


def find_all(input_dir, postfix="", return_dict=False):
    names = []
    for root, _, files in os.walk(input_dir, topdown=False):
        for name in files:
            if name.endswith(postfix):
                names.append(os.path.join(root, name))

    names = sorted(names)

    if return_dict:
        names = {os.path.basename(name): name for name in names}

    return names


def normalize_obs(x, max_tp=60):
    x = (x - obs_mean) / obs_std

    tp = x[:, -1] 
    tp = np.log(1 + tp.clip(min=0, max=max_tp))     
    x[:, -1] = tp   
    
    return x


def normalize_wrf(x, max_tp=60):
    x = (x - wrf_mean) / wrf_std
    
    tp = x[:, -1] 
    tp = tp[1:] - tp[:-1]
    tp = np.concatenate([x[[0], -1], tp])
    tp = np.log(1 + tp.clip(min=0, max=max_tp))    
    
    x[:, -1] = tp    
    return x



def convert_to_nc(img, init_time, latlon, degree):
    nt = len(img)
    lat1, lat2, lon1, lon2 = latlon
    lats = np.arange(lat1, lat2+degree, degree).astype(np.float32)
    lons = np.arange(lon1, lon2+degree, degree).astype(np.float32)
    time = np.linspace(1, nt, nt).astype(np.int32)

    ds = xr.Dataset(
        {
            "pre": (["time", "lat", "lon"], img, {"units": "mm"})
        },
        coords={
            "lon": lons,
            "lat": lats,
            "time": time,
            "reference_time": init_time
        },
    )
    return ds



def match_wrf(init_time, delay=6, interval=3, prefix='PWAFS', postfix='0000.nc'):
    wrf_init = init_time - pd.Timedelta(hours=delay)
    wrf_init = wrf_init.strftime("%Y%m%d%H")

    hour = int(wrf_init[-2:]) // interval * interval
    wrf_init = wrf_init[:-2] + f'{hour:02d}'

    wrf_name = f'{prefix}_{wrf_init}{postfix}'
    wrf_init = pd.to_datetime(wrf_init, format="%Y%m%d%H")
    diff = (init_time - wrf_init) / pd.Timedelta(hours=1)

    wrf_path = os.path.join(wrf_dir, wrf_name)
    return wrf_path, wrf_init, int(diff)


def load_wrf(path, diff, input_frames, future_frames, fmt="%Y%m%d%H"):

    if not os.path.exists(path):
        print(f'{path} do not exist!')
        return None

    data = xr.open_dataset(path).fillna(0)

    wrf_imgs = []
    for name in wrf_names['pl']:
        if name not in data:
            print(f'{name} not exist')
            return None
        
        src_level = data.VV.level.values
        
        for lvl in wrf_levels:
            if lvl not in src_level:
                print(f'{lvl} not in {src_level}')
                return None
            
            x = data[name].sel(level=lvl).values
            wrf_imgs.append(x)
    
    
    for name in wrf_names['sfc']:
        if name not in data:
            print(f'{name} not exist')
            return None
        
        x = data[name].values
        wrf_imgs.append(x)

    wrf_imgs = np.stack(wrf_imgs, axis=1)
    wrf_imgs = wrf_imgs[diff-input_frames:diff+future_frames]
    wrf_imgs = normalize_wrf(wrf_imgs)
    return wrf_imgs


def check_wrf(wrf, total_frames):
    if wrf is None:
        return False 

    if wrf.shape[0] != total_frames:
        print(f"wrf has wrong shape: {wrf.shape}!")
        return False
    
    if np.abs(wrf).max() > 100:
        print(f"wrf has abnormal value: {wrf.max()}!")
        return False

    return True



def match_obs(init_time, postfix='.GRB2'):
    
    dates = []
    t1 = init_time - pd.Timedelta(hours=input_frames)
    for time in pd.date_range(t1, init_time, freq="1h", inclusive="left"):
        time = time.strftime("%Y%m%d")
        dates.append(time)
    dates = np.unique(dates)

    src_paths = []
    dst_paths = {}

    for date in dates:
        # TODO, modify dir_name to yours
        tem_dir = os.path.join(obs_dir, 'tem', date)
        win_dir = os.path.join(obs_dir, 'win', date)
        pre_dir = os.path.join(obs_dir, 'pre', date)   
        src_paths += find_all(tem_dir, postfix=postfix)  
        src_paths += find_all(win_dir, postfix=postfix)  
        src_paths += find_all(pre_dir, postfix=postfix)  

    for path in src_paths:
        name = os.path.basename(path)
        name = name.replace(postfix, '')
        time = name[-10:]                 

        if time not in dst_paths:
            dst_paths[time] = {}

        if 'HOR-TEM' in name and 'TEMQC' not in name:
            dst_paths[time]['tem'] = path                
        
        if 'HOR-WIN' in name and 'WINQC' not in name:
            dst_paths[time]['win'] = path        

        if 'HOR-PRE' in name and 'PREQC' not in name:
            dst_paths[time]['pre'] = path                


    obs_paths = []
    t1 = init_time - pd.Timedelta(hours=input_frames)
    for time in pd.date_range(t1, init_time, freq="1h", inclusive="left"):
        time = time.strftime("%Y%m%d%H")
        obs_path = dst_paths.get(time, {})
        obs_paths.append(obs_path)

    return obs_paths



def read_grib(path):
    try:
        data = pygrib.open(path).select()
    except:
        print(f"read grib failed: {path}")
        return None       
    return data



def load_obs(obs_paths):

    obs_imgs = []

    for path in obs_paths:
        tem_name = path.get('tem', '')
        win_name = path.get('win', '')
        pre_name = path.get('pre', '')

        if not os.path.exists(tem_name):
            return None 

        if not os.path.exists(win_name):
            return None 

        if not os.path.exists(pre_name):
            return None 

        tem = read_grib(tem_name)
        win = read_grib(win_name)
        pre = read_grib(pre_name)

        if tem is None or win is None or pre is None:
            return None

        lat1, lat2, lon1, lon2 = obs_latlon

        try:
            t2m, _, _ = tem[0].data(
                lat1=lat1,lat2=lat2,
                lon1=lon1,lon2=lon2,
            )
        except:
            print('crop tem to small region')
            return None
        
        try:
            u10, _, _ = win[0].data(
                lat1=lat1,lat2=lat2,
                lon1=lon1,lon2=lon2,
            )
            u10_mask = u10.mask
            u10 = u10.data

            v10, _, _ = win[1].data(
                lat1=lat1,lat2=lat2,
                lon1=lon1,lon2=lon2,
            )
            u10[u10_mask] = 0
            v10[u10_mask] = 0
        except:
            return None

        try:
            tp, _, _ = pre[0].data(
                lat1=lat1,lat2=lat2,
                lon1=lon1,lon2=lon2,
            )
            tp_mask = tp.mask
            tp = tp.data
            tp[tp_mask] = 0
        except:
            return None

        
        obs = np.stack([t2m, u10, v10, tp])
        obs_imgs.append(obs)
    
    
    obs_imgs = np.array(obs_imgs, dtype=np.float32)
    obs_imgs = normalize_obs(obs_imgs)
    return obs_imgs



def check_obs(obs):
    
    if obs is None:
        return False

    if obs.shape[0] != input_frames:
        return False 

    return True



def process_data(obs, wrf):
    img_h, img_w = obs.shape[-2:]
    obs_imgs = np.zeros((obs.shape[:2]+tuple(input_size)), dtype=np.float32)
    obs_imgs[:, :, :img_h, :img_w] = obs
    obs_imgs = obs_imgs[None, :, 1:]
    print(f"obs_imgs: {obs_imgs.shape}, minmax: {obs_imgs.min():.3f} ~ {obs_imgs.max():.3f}")

    y1, y2, x1, x2 = wrf_rect
    wrf_imgs = np.zeros((wrf.shape[:2]+tuple(input_size)), dtype=np.float32)
    wrf_imgs[:, :, y1:y2, x1:x2] = wrf
    wrf_imgs = wrf_imgs[None]
    print(f"wrf_imgs: {wrf_imgs.shape}, minmax: {wrf_imgs.min():.3f} ~ {wrf_imgs.max():.3f}")

    inputs = dict(
        obs_imgs=obs_imgs,
        wrf_imgs=wrf_imgs,
    )
    return inputs


def process_output(output, max_tp=60):
    output = output * obs_std[1:] + obs_mean[1:]

    tp = output[:, -1] 
    tp = np.exp(tp) - 1
    tp = tp.clip(min=0, max=max_tp)    
    output[:, -1] = tp   
    
    u10 = output[:, 0]
    v10 = output[:, 1]
    prec= output[:, 2]
    wind = np.sqrt(u10 ** 2 + v10 **2)
    
    results = {
        'u10': u10,
        'v10': v10,
        'prec': prec,
        'wind': wind,
    }
    return results



def main():

    fmt="%Y%m%d%H"    
    future_frames = args.future_frames + 1

    start_time = pd.to_datetime('2022072602', format=fmt)
    end_time = pd.to_datetime('2022072602', format=fmt)

    # initial model only once
    # session = onnxruntime.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    print('load model ...')        
    start = time.perf_counter()
    sessions = []
    for i, name in enumerate(args.models):
        model_name = os.path.join(model_dir, name)
        print(f' {i}: {model_name}')
        session = onnxruntime.InferenceSession(model_name, providers=['CPUExecutionProvider'])
        sessions.append(session)
    load_time = time.perf_counter() - start
    print('load_time: {:.3f} sec'.format(load_time))        
    
    # session = onnxruntime.InferenceSession(args.model, providers=["CUDAExecutionProvider"])

    for init_time in pd.date_range(start_time, end_time, freq="6h"):
        time_str = init_time.strftime(fmt)
        assert time_str[-2:]  in ['02', '08', '14', '20'] 
        
        start = time.perf_counter()

        wrf_time = convert_time(init_time, 'bj2utc')
        wrf_path, wrf_init, diff = match_wrf(wrf_time) 
        wrf_init = convert_time(wrf_init, 'utc2bj')
        print(f'init_time: {init_time} BJT, wrf_init: {wrf_init} BJT, diff: {diff}h')
        
        wrf = load_wrf(wrf_path, diff, input_frames, future_frames)

        if not check_wrf(wrf, input_frames+future_frames):
            continue

        
        obs_paths = match_obs(init_time) 
        obs = load_obs(obs_paths)

        if not check_obs(obs):
            continue
        
        inputs = process_data(obs, wrf)
        data_time = time.perf_counter() - start
        print('date_time: {:.3f} sec'.format(data_time))
        
        start = time.perf_counter()
              
        outputs = []
        for session in sessions:
            output = session.run(None, inputs)
            outputs.append(output)
        outputs = np.mean(outputs, 0)
              
        results = process_output(outputs[0][0])
        test_time = time.perf_counter() - start
        print('test_time: {:.3f} sec'.format(test_time))
        
        prec = results['prec'][1:]
        y1, y2, x1, x2 = out_rect
        prec = prec[:, y1:y2, x1:x2]    
        
        
        for t, img in enumerate(prec, 1):
            title = f'{time_str}_{t:02d}'
            print('time: {:02d}, shape: {}, max: {:.3f}'.format(t, img.shape, img.max()))
            plt.imshow(img, 'jet')
            plt.title(title)
            plt.axis("off")            
            save_name = os.path.join(args.save_dir, f'{title}.jpg')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0.0, transparent='true', dpi=100)
        
    
        prec = convert_to_nc(prec, init_time, out_latlon, out_degree)
        
        



if __name__ == "__main__":
    main()

