# scripts/check.py
import arviz as az

def main():
    idata = az.from_netcdf(r".\experiments\runs\exp_baseline_20250911-133337\posterior.nc")
    print("groups:", idata.groups())
    print("prior vars:", list(getattr(idata, "prior", {}).keys()) if hasattr(idata, "prior") else None)
    print("posterior vars:", list(idata.posterior.data_vars))

if __name__ == "__main__":
    main()
