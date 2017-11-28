import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

dices = [0.94878744650499292, 0.96349431818181819, 0.96607270135424095, 0.94848219497956798, 0.92243346007604565, 0.95611644916104976, 0.95877318116975752, 0.96696568418054962, 0.96407359119223524, 0.91703794745736678, 0.96193622522263722, 0.96428061151593081, 0.96377749029754201, 0.94654322831118409, 0.90896423594983744, 0.9565655112900906, 0.96388809591778479, 0.95646514318773501, 0.9256877990430622, 0.91284967722102672, 0.8999377528789293, 0.93607784431137719, 0.94722016308376578, 0.93565243356225736, 0.92968631610850128]

dices = np.array(dices)*100
dices = dices.reshape((5,5))
print(dices.shape)
print(dices)

# ValueError: Colormap red is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

ax = sns.heatmap(dices, annot=True, cmap='OrRd', xticklabels=[-10, -5, 0, 5, 10], fmt='.2f', yticklabels=[-10, -5, 0, 5, 10]) # YlGnBu
fig = ax.get_figure()
fig.savefig('out.eps')