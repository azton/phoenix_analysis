
import json, glob
import numpy as np
import matplotlib.pyplot as plt


rpath = '/mnt/c/Users/azton/Projects/phoenix_analysis/p3_region_logs/phx256-1'
rfiles = glob.glob('%s/*'%(rpath))

with open(rfiles[0], 'r') as f:
    data = json.load(f)

print(data['post_state'].keys())
print(type(data['post_state']['p2_stars']), len(data['post_state']['p2_stars']))


p2s = []
p2form = []
dts = []
high_p2_pids = []
neg_dt = []
dt_cnt = [[],[]]
p2mass = []
p3mass = []
nop2_cnt = 0
all_cnt = 0
prior_p2 = 0
nopostp3 = 0
for rfile in rfiles:
    with open(rfile, 'r') as f:
        data = json.load(f)
    np3 = len(data['post_state']['p3_stars']) + len(data['post_state']['p3_remnants']) + len(data['post_state']['p3_bh'])
    npriorp2 = len(data['prior_state']['p2_formation_time'])
    if np3 == 0:
        nopostp3 += 1
        print('No P3s: %s'%rfile)
    if npriorp2:
        prior_p2 += 1
    if np3 > 0 and npriorp2 == 0:
        all_cnt += 1
        cur_time = data['post_state']['current_time']
        formed_p2 = False
        for p2time in data['post_state']['p2_formation_time']:
            if p2time > current_time and p2time < current_time+16.8:
                formed_p2 = True
        if not formed_p2: nop2_cnt += 1
        dts += [data['post_state']['current_time'] - data['prior_state']['current_time']]
        # if dt < 30:
        p2s += [len(data['post_state']['p2_stars'])] 
        p2form += [t - data['prior_state']['current_time'] 
                    if data['post_state']['p2_stars'][i] not in data['prior_state']['p2_stars'] else -1 
                    for i, t in enumerate(data['post_state']['p2_formation_time'])]
        p3mass += [
            sum(data['post_state']['p3_masses'])]
        p2mass += [sum(data['post_state']['p2_masses']) - sum(data['prior_state']['p2_masses'])]
        if len(data['post_state']['p2_stars']) > 50:
            high_p2_pids.append(
                int(os.path.split(rfile)[-1].split('_')[-1].split('.')[0])
            )

        if np.any(np.array([t - ftime for t in data['post_state']['p2_formation_time']]) < 0) and ftime > 0:
            neg_dt.append(
                int(os.path.split(rfile)[-1].split('_')[-1].split('.')[0])
            )
            dt_cnt[0].append(min([t - data['prior_state']['current_time'] for t in data['post_state']['p2_formation_time']]))
            dt_cnt[1].append(len(data['post_state']['p2_stars']))

print('n_prior_p2:',prior_p2, 
        '\nn_nop2: ', nop2_cnt, 
        '\nn_nop3 in post: ',nopostp3, 
        '\nall qualifying regions: ',all_cnt)
print(high_p2_pids)
print(neg_dt)


fig, ax = plt.subplots(5, figsize=(12,18))

p2s = np.array(p2s)
np2forms = (p2s[p2s == 0]).size
nfew = (p2s[p2s < 5]).size
nlots = (p2s[p2s < 15]).size
p2form = np.array(p2form)
p2mass = np.array(p2mass)[np.array(p2mass) > 1]
n_neg = p2form[p2form < 0].size
nop3chng = (np.array(p3mass)[np.array(p3mass) == 0]).size

ax[0].hist(p2s, bins=25)
ax[1].hist(p2form, bins=25, range=[0,20], cumulative = True, histtype='step', density=True)
# ax[1].set_yscale('log')
ax[2].scatter(dt_cnt[0], dt_cnt[1])
ax[3].hist(p2mass, bins=25)
ax[3].set_yscale('log')
# ax3t = ax[3].twiny()
hist, p3bins, _ = ax[4].hist(p3mass, bins=100, range=[0,500], alpha = 0.5, color='tab:blue', density=True)
p3mass = np.array(p3mass)
p3nop2 = p3mass[p2s == 0]
ax[4].hist(p3nop2, bins=p3bins, alpha=0.5, color='tab:orange', density=True)
ax[4].set_yscale('log')

ax[4].set_title('%d no change'%nop3chng)
ax[0].set_yscale('log')
ax[0].set_title('$N_{!P2} = %d, %d, %d$'%(np2forms, nfew, nlots))
ax[1].set_title('$N_{neg} = %d/%d$'%(len(neg_dt), len(p2form)))

plt.savefig('images/cluster_regions_analysis.pdf')


print(p2form[p2form > 0].size, p2form[p2form == 0].size)


# plotting p3 mass as fn of time in regions:
meanvals = np.zeros((30))
fig, ax = plt.subplots(figsize=(4,4))
m_samples = [] # list of all bin p3 masses for all samples
n_samples = [] # list of all bin p3 counts
for rfile in rfiles:
    timebins = np.linspace(0, 17, 30)
    binvals = np.zeros((31))
    cntvals = np.zeros((31))
    with open(rfile, 'r') as f:
        data = json.load(f)

    tnow = data['prior_state']['current_time']
    tbins = np.digitize([t-tnow for t in data['post_state']['p3_formation_time']], timebins)
    for i, t in enumerate(tbins):
        binvals[t] += data['post_state']['p3_masses'][i]
        cntvals[t] += 1

    tbins = np.digitize([t-tnow for t in data['post_state']['remnant_birth']], timebins)
    for i, t in enumerate(tbins):
        binvals[t] += data['post_state']['remnant_masses'][i]
        cntvals[t] += 1
    # binvals = np.array([binvals[:i].sum() for i in range(1, binvals.size)])

    tbins = np.digitize([t-tnow for t in data['post_state']['p3_bh_birth']], timebins)
    for i, t in enumerate(tbins):
        binvals[t] += data['post_state']['p3_bh_masses'][i]
        cntvals[t] += 1
    binvals = np.array([binvals[:i].sum() for i in range(1, binvals.size)])
    cntvals = np.array([cntvals[:i].sum() for i in range(1, cntvals.size)])
    n_samples += [cntvals]
    m_samples += [binvals]
    # meanvals += binvals
    # ax.plot(timebins, binvals, alpha=0.1, color='tab:blue')
samples = np.array(m_samples)
mean_samples = samples.mean(0)
std_samples = samples.std(0)
ax.plot(timebins, mean_samples, alpha=1.0, color='tab:blue')
ax.plot(timebins, mean_samples + std_samples, alpha=1.0, color='tab:green')
ax.plot(timebins, mean_samples - std_samples, alpha = 1.0, color='tab:green')
ax.fill_between(timebins, mean_samples, mean_samples+std_samples, color='tab:green', alpha=0.1)
ax.fill_between(timebins, mean_samples, mean_samples-std_samples, color='tab:green', alpha=0.1)

axt = ax.twinx()
samples = np.array(n_samples)
mean_samples = samples.mean(0)
std_samples = samples.std(0)
axt.plot(timebins, mean_samples, alpha=1.0, color='tab:blue',linestyle='--')
axt.plot(timebins, mean_samples + std_samples, alpha=1.0, color='tab:green', linestyle='--')
axt.plot(timebins, mean_samples - std_samples, alpha = 1.0, color='tab:green', linestyle='--')
axt.fill_between(timebins, mean_samples, mean_samples+std_samples, color='tab:green', alpha=0.1)
axt.fill_between(timebins, mean_samples, mean_samples-std_samples, color='tab:green', alpha=0.1)
plt.savefig('images/p3_massVtime.pdf')


## Plot the mass distribution of regions that a: have p2 stars
                                            # b: have no p2 stars
                                        
withp2_mass = []
nop2_mass = []
for rfile in rfiles:
    with open(rfile, 'r') as f:
        data = json.load(f)
    if len(data['post_state']['p2_masses']) > 0:
        withp2_mass += data['post_state']['p3_masses']
        withp2_mass += data['post_state']['remnant_masses']
        withp2_mass += data['post_state']['p3_bh_masses']
    else:
        nop2_mass += data['post_state']['p3_masses']
        nop2_mass += data['post_state']['remnant_masses']
        nop2_mass += data['post_state']['p3_bh_masses']
fig, ax = plt.subplots(figsize=(4,4))
# bins = np.array([i for i in range(11,41)] + [i for i in range(140, 261, 5)])
ax.hist([withp2_mass, nop2_mass], bins=25, label = ['P2','No P2'], density=True)
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('$M_*~[M_\odot]$')
ax.set_ylabel('d$N$/d$M$')
plt.savefig('images/p2p3_mass_config.pdf', bbox_inches='tight')






