import datetime
import os

import dotenv
import xarray

dotenv.load_dotenv()

from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds
from earth2mip.networks import get_model
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

print("Fetching Pangu model package...")
package = registry.get_model("e2mip://pangu")

print("Fetching DLWP model package...")
package = registry.get_model("e2mip://dlwp")

print("Fetching FCNv2 model package...")
package = registry.get_model("e2mip://fcnv2_sm")

print("Fetching graphcast_operational model package...")
package = registry.get_model("e2mip://graphcast_operational")

import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu
import earth2mip.networks.fcnv2_sm as fcnv2_sm
# import earth2mip.networks.graphcast_operational as graphcast_operational

# Output directoy
output_dir = "outputs/02_model_comparison"
os.makedirs(output_dir, exist_ok=True)

print("Loading models into memory")
# Load DLWP model from registry
package = registry.get_model("dlwp")
dlwp_inference_model = dlwp.load(package)

# Load Pangu model(s) from registry
package = registry.get_model("pangu")
pangu_inference_model = pangu.load(package)

# Load DLWP model from registry
package = registry.get_model("fcnv2_sm")
fcnv2_sm_inference_model = fcnv2_sm.load(package)

# Load DLWP model from registry
package = registry.get_model("graphcast_operational")
# graphcast_operational_inference_model = graphcast_operational.load(package)

time = datetime.datetime(2018, 1, 1)

# DLWP datasource
dlwp_data_source = cds.DataSource(dlwp_inference_model.in_channel_names)

# Pangu datasource, this is much simplier since pangu only uses one timestep as an input
pangu_data_source = cds.DataSource(pangu_inference_model.in_channel_names)

# Pangu datasource, this is much simplier since pangu only uses one timestep as an input
fcnv2_sm_data_source = cds.DataSource(fcnv2_sm_inference_model.in_channel_names)

# # Pangu datasource, this is much simplier since pangu only uses one timestep as an input
# graphcast_operational_data_source = cds.DataSource(graphcast_operational_inference_model.in_channel_names)

print("Running Pangu inference")
pangu_ds = inference_ensemble.run_basic_inference(
    pangu_inference_model,
    n=24,  # Note we run 24 steps here because Pangu is at 6 hour dt (6 day forecast)
    data_source=pangu_data_source,
    time=time,
)
pangu_ds.to_netcdf(f"{output_dir}/pangu_inference_out.nc")
print(pangu_ds)

print("Running DLWP inference")
dlwp_ds = inference_ensemble.run_basic_inference(
    dlwp_inference_model,
    n=24,  # Note we run 24 steps. DLWP steps at 12 hr dt, but yeilds output every 6 hrs (6 day forecast)
    data_source=dlwp_data_source,
    time=time,
)
dlwp_ds.to_netcdf(f"{output_dir}/dlwp_inference_out.nc")
print(dlwp_ds)

print("Running FCNv2_sm inference")
fcnv2_sm_ds = inference_ensemble.run_basic_inference(
    fcnv2_sm_inference_model,
    n=24,  # Note we run 24 steps. DLWP steps at 12 hr dt, but yeilds output every 6 hrs (6 day forecast)
    data_source=fcnv2_sm_data_source,
    time=time,
)
fcnv2_sm_ds.to_netcdf(f"{output_dir}/fcnv2_sm_inference_out.nc")
print(fcnv2_sm_ds)

graphcast_operational_inference_model = get_model("e2mip://graphcast_operational", device="cuda:0")
graphcast_operational_data_source = cds.DataSource(graphcast_operational_inference_model.in_channel_names)


print("Running graphcast_operational inference")
graphcast_operational_ds = inference_ensemble.run_basic_inference(
    graphcast_operational_inference_model,
    n=24,  # Note we run 24 steps. DLWP steps at 12 hr dt, but yeilds output every 6 hrs (6 day forecast)
    data_source=graphcast_operational_data_source,
    time=time,
)
graphcast_operational_ds.to_netcdf(f"{output_dir}/graphcast_operational_inference_out.nc")
print(graphcast_operational_ds)
