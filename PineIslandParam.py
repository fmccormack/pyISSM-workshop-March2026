import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
import pyissm
import ccdtools as dp
import xarray as xr

print("-------------------------------------------------------------"  )
print(f" RUNNING PineIslandParam.py PARAMETERIZATION SCRIPT"  )
print("-------------------------------------------------------------\n")

# Set-up data catalogue
catalog = dp.catalog.DataCatalog()

print('-- Setting ice mask...')
# Initialise levelset fields to all ice and no ocean
md.mask.ice_levelset  = -1 * np.ones(md.mesh.numberofvertices)
md.mask.ocean_levelset = np.ones(md.mesh.numberofvertices)

# Interpolate BedMachine v3 Ice Mask
bedmachine_data = catalog.load_dataset('measures_bedmachine_antarctica', version='v3')
mask = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'mask', md.mesh.x, md.mesh.y, interpolation_type='nearest')

# Set ocean areas (mask == 0) to positive levelset (no ice)
print('  - Remove ice from ocean areas...')
no_ice_mask = (mask < 1)
md.mask.ice_levelset[no_ice_mask] = 1

# Set floating ice (mask == 3) and ocean (mask == 0) to negative levelset (ocean)
print('  - Defining ocean levelset...')
ocean_mask = (mask == 3) | (mask == 0)
md.mask.ocean_levelset[ocean_mask] = -1

print('-- Setting geometry...')
md.geometry.bed       = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'bed',       md.mesh.x, md.mesh.y, interpolation_type='nearest')
md.geometry.surface   = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'surface',   md.mesh.x, md.mesh.y, interpolation_type='nearest')
md.geometry.thickness = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'thickness', md.mesh.x, md.mesh.y, interpolation_type='nearest')
md.geometry.base = md.geometry.surface - md.geometry.thickness

print('  - Correcting surface...')
md.geometry.surface   = np.maximum(md.geometry.surface, 1)
md.geometry.thickness = md.geometry.surface - md.geometry.base

print('  - Adjusting thickness...')
md.geometry.thickness = np.maximum(md.geometry.thickness, 10)
md.geometry.base      = md.geometry.surface - md.geometry.thickness
pos = md.geometry.base < md.geometry.bed
md.geometry.bed[pos]  = md.geometry.base[pos]

print('  - Setting ice shelf base using hydrostatic equilibrium...')
floating_base = md.materials.rho_ice / (md.materials.rho_ice - md.materials.rho_water) * md.geometry.surface
pos = (floating_base > md.geometry.bed) & (md.mask.ocean_levelset < 0)
md.geometry.base[pos] = floating_base[pos]
di = md.materials.rho_ice / md.materials.rho_water
md.geometry.thickness  = md.geometry.surface - md.geometry.base
md.mask.ocean_levelset = md.geometry.thickness + md.geometry.bed / di

pos = (md.geometry.base < md.geometry.bed) | (md.mask.ocean_levelset > 0)
md.geometry.bed[pos] = md.geometry.base[pos]

md.groundingline.intrusion_distance = 0

print('  - Adjusting ice mask...')
# Offset the mask by one element so that there are no cliffs at the transition
elems = md.mesh.elements - 1
vals  = md.mask.ice_levelset[elems]
pos_e = np.max(vals, axis=1) > 0
md.mask.ice_levelset[elems[pos_e, :]] = 1
pos = (md.mask.ice_levelset < 0) & (md.geometry.surface < 0)
md.mask.ice_levelset[pos] = 1
pos = (md.mask.ice_levelset < 0) & np.isnan(md.geometry.surface)
md.mask.ice_levelset[pos] = 1

print('-- Setting velocity...')
velocity_data = catalog.load_dataset('measures_insar_based_antarctica_ice_velocity_map', version='v2')
md.inversion.vx_obs = pyissm.data.interp.xr_to_mesh(velocity_data, 'VX', md.mesh.x, md.mesh.y)
md.inversion.vy_obs = pyissm.data.interp.xr_to_mesh(velocity_data, 'VY', md.mesh.x, md.mesh.y)
pos = np.isnan(md.inversion.vx_obs) | np.isnan(md.inversion.vy_obs)
md.inversion.vx_obs[pos] = 0
md.inversion.vy_obs[pos] = 0
md.inversion.vel_obs   = np.sqrt(md.inversion.vx_obs**2 + md.inversion.vy_obs**2)
md.initialization.vx   = md.inversion.vx_obs.copy()
md.initialization.vy   = md.inversion.vy_obs.copy()
md.initialization.vz   = np.zeros(md.mesh.numberofvertices)
md.initialization.vel  = md.inversion.vel_obs.copy()

print('-- Setting geothermal heat flux (Staal)...')  # W/m^2
geothermal_data = catalog.load_dataset('antarctic_geothermal_heat_flow_model_aq1', compat='override')
geothermal_data = geothermal_data.chunk({'X': -1, 'Y': -1})
geothermal_data['Q'] = geothermal_data['Q'].interpolate_na(dim='X', method='linear', fill_value='extrapolate')
geothermal_data['Q'] = geothermal_data['Q'].interpolate_na(dim='Y', method='linear', fill_value='extrapolate')

ghf = pyissm.data.interp.xr_to_mesh(
    geothermal_data, 'Q', md.mesh.x, md.mesh.y, x_var='X', y_var='Y', interpolation_type='bilinear')
ghf_smooth = pyissm.tools.interp.averaging(md, ghf, 5)[:,0]
ind = np.where(np.isnan(ghf_smooth)==1)[0]
ghf_smooth[ind] = griddata((md.mesh.x[~ind],md.mesh.y[~ind]), ghf_smooth[~ind], (md.mesh.x[ind], md.mesh.y[ind]), method='nearest')
ind = np.where(np.isnan(ghf_smooth)==1)[0]
ghf_smooth[ind]=0
md.basalforcings.geothermalflux = ghf_smooth

print('-- Setting floating ice basal melt rate (Paolo et al.)...')
melt_data = catalog.load_dataset('measures_its_live_antarctic_quarterly_ice_shelf_height_change', version='v1')
melt_mean = pyissm.data.interp.xr_to_mesh(melt_data, 'melt_mean', md.mesh.x, md.mesh.y)
md.basalforcings.floatingice_melting_rate = -1*melt_mean

print('-- Initialising friction fields...')
md.groundingline.intrusion_distance = 0
md.friction.coefficient = np.zeros(md.mesh.numberofvertices)
md.friction.p           = np.ones(md.mesh.numberofelements)
md.friction.q           = np.ones(md.mesh.numberofelements)
