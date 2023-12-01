# %% ------------------------- INSTALL PACKAGES ------------------------
# COMMENT IF NOT NEEDED

!pip install ruptures
!pip install scipy
!pip install statsmodels
!pip install pandas
#%% -------------------------- LOAD ALL PACKAGES -----------------------
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import statsmodels.api as sm
from scipy import stats
import pandas as pd
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.discrete.diagnostic import PoissonDiagnostic
import json

# %% - FUNCTION (single peak)

def get_estimates(y, t, lag=10, b_hat = -1, p_value = 0.975, num_peak = 2, duration_index = 100, bck_duration_index = 200,padding = 10, filename=""):
	plt.plot(t,y)
	plt.title(f"noisy measurement: {filename}")
	plt.xlabel("time")
	plt.ylabel("count")
	plt.ylim(0,500)
	plt.show()
    # ---------- CHANGE POINT DETECTION --------------
    # To detect the point of maximum count before falling
    # And to detect the point where there is an increasing count
	y_dif = np.array([y[i]-y[i-(lag+1)] for i in range(lag, len(y))])
	model = "l1"  # "l1", "rbf", "linear", "normal", "ar"
	algo = rpt.BottomUp(model=model).fit(y_dif)
	my_bkps = algo.predict(n_bkps=1) # get T_hat_max
	T_hat_max = my_bkps[0]
	rpt.show.display(y_dif, my_bkps, figsize=(10, 6))
	plt.title(f"Change point of diffrence between Poisson: {filename}")
	plt.ylim(min(y_dif), max(y_dif))
	plt.xlabel("time")
	plt.ylabel("count dif")
	plt.show()
	model = "l1"  # "l1", "rbf", "linear", "normal", "ar"
	algo = rpt.BottomUp(model=model).fit(y_dif)
	my_bkps = algo.predict(n_bkps=num_peak) # get T_hat_max
	rpt.show.display(y_dif, my_bkps, figsize=(10, 6))
	plt.ylim(min(y_dif), max(y_dif))
	plt.show()
	T_hat_min = my_bkps[0]
	rpt.show.display(y, my_bkps, figsize=(10, 6))
	plt.title(f"Estimated rise time: {filename}")
	plt.ylim(0, 300)
	plt.show()
	#print([T_hat_min-padding, T_hat_max+padding])
	# ------------ SPLIT DATA --------------------------
	# Split data before the rise time and split data after it has reach
	# maximum
	begin = T_hat_max+padding
	end = T_hat_max + padding + duration_index
	if duration_index > len(t) - T_hat_max + padding or duration_index < 0:
		end = len(t)- T_hat_max - padding
	y_1 = y[:T_hat_min-padding]
	y_2 = np.array(y[begin:end])
	y_3 = np.array(y[-bck_duration_index:])
	b_2_hat = np.mean(y_3)
	# Estimate background activity if non is given.
	if b_hat == -1:
		b_hat = np.mean(y_1)
	b_1_upper = b_hat + stats.norm.ppf(p_value)*np.sqrt(b_hat)
	b_1_lower = b_hat - stats.norm.ppf(p_value)*np.sqrt(b_hat)
	b_2_upper = b_2_hat + stats.norm.ppf(p_value)*np.sqrt(b_2_hat)
	b_2_lower = b_2_hat - stats.norm.ppf(p_value)*np.sqrt(b_2_hat)
	system_matrix = np.array([np.array(t[begin:end]) - t[begin], np.repeat(1, len(y_2))])
	system_matrix = np.transpose(system_matrix)
	res = Poisson(np.array(y_2)-b_2_hat, system_matrix).fit(disp=0)
	system_matrix = np.array([np.array(t[begin:]) - t[begin], np.repeat(1, len(t[begin:]))])
	system_matrix = np.transpose(system_matrix)
	pred = res.get_prediction(system_matrix)
	param = res.params
	#print(param)
	#frame = pred.summary_frame(alpha=p_value)
	# Get confidence interval for parameter if p_values are given
	confi = res.conf_int(alpha=p_value, cols=None)
	lambda_ci_lower,lambda_ci_upper = confi[0][0], confi[0][1]
	log_N0_ci_lower,log_N0_ci_upper = confi[1][0], confi[1][1]
	#print(f"confidence interval: {confi}")
 
	# Get fitted values
	t_1 = t[:T_hat_min-padding]
	t_2 = t[begin:]
	# predicted
	y_2_hat = np.array(pred.predicted) + b_2_hat
	y_1_hat = np.repeat(b_hat, len(t[:T_hat_min-padding]))
	# Lower confidence interval
	y_1_hat_lower = np.repeat(b_1_lower, len(t[:T_hat_min-padding]))
	y_1_hat_upper = np.repeat(b_1_upper, len(t[:T_hat_min-padding]))
	# highest N0 + lowest lambda
	y_2_hat_upper = np.exp(log_N0_ci_upper )*np.exp(lambda_ci_upper*(np.array(t_2)-t[begin])) + b_2_upper
	# lowest N0 + highest lambda
	y_2_hat_lower = np.exp(log_N0_ci_lower )*np.exp(lambda_ci_lower*(np.array(t_2)-t[begin])) + b_2_lower
	# Get confidence interval for fitted value
	#y_2_hat_ci_upper = np.array(frame.obs_ci_upper)
	#y_2_hat_ci_lower = np.array(frame.obs_ci_lower)
	# Plot all in 1 graph
	plt.plot(y)
	plt.axvspan(0, t[T_hat_min-padding], 											color='g', alpha=0.5, lw=0, label=f"bgr time: {[t[0], t[begin]]}")
	plt.axvspan(t[T_hat_min-padding], 		t[T_hat_max+padding], 					color="y", alpha = 0.5, lw=0, label=f"rise time: {[t[T_hat_min], t[T_hat_max]]}")
	plt.axvspan(t[begin], t[end], 													color='orange', alpha=0.5, lw=0, label=f"fitting duration: {[t[begin], t[end]]}")
	plt.axvspan(t[T_hat_max+padding], max(t), 										color='r', alpha=0.5, lw=0, label=f"bgr + tracer time: {[t[begin+duration_index], t[-1]]}")
	plt.hlines(b_hat, t[0], t[T_hat_max+padding], 									colors="orange", linestyle='--')
	plt.hlines(b_2_hat, t[T_hat_max+padding], t[len(t)-1], 							colors="orange", linestyle='--', label=f'bgr mean: {np.round(b_hat,2), np.round(b_2_hat,2)}')
	plt.plot(np.concatenate([t_1, t_2]), np.concatenate([y_1_hat, y_2_hat]), 		color='yellow', linestyle='--', label="fitted mean")
	plt.plot(np.concatenate([t_1, t_2]), np.concatenate([y_1_hat_lower, y_2_hat_lower]), color='purple', linestyle='--', label=f"Confi.Int at {p_value}")
	plt.plot(np.concatenate([t_1, t_2]), np.concatenate([y_1_hat_upper, y_2_hat_upper]), color='purple', linestyle='--')
	plt.ylim(0, 500)
	plt.legend(loc='upper right')
	plt.ylabel("count")
	plt.xlabel("time")
	plt.title(f"fitted estimates: {filename}, max at {[T_hat_max, np.round(y_2_hat[0],2)]}, lambda: {np.round(-param[0],5)}")
	plt.show()
	output = {
		"parameters": [b_hat, b_2_hat, param[0], np.exp(param[1])],
		"rise_time": [t[T_hat_min], t[T_hat_max]],
		"max_loc": [t[T_hat_max], y_2[0]],
		"fitted": np.concatenate([y_1_hat, y_2_hat]),
		"upper_ci": np.concatenate([y_1_hat_upper, y_2_hat_upper]),
		"lower_ci": np.concatenate([y_1_hat_lower, y_2_hat_lower])
	}
	print(res.summary())
	return output


# %% --------------------------- USER INPUT ---------------------------
filename = "9b.crv"										# REQUIRED: FILENAME			
column = 6												    # OPTIONAL: WHICH COLUMN TO USE: DEFAULT 6
# %% --------------------------- END USER INPUT -----------------------
data = np.loadtxt(filename)
coincidence = data[:,column]	
y_data = np.array(coincidence)
t = np.arange(0,len(y_data))
# %% --------------------- HYPER PARAMETERS & FITTING ------------------
output = get_estimates(y=y_data, 			      # Count Data					, AUTO
                       t=t, 				        # Time 							, AUTO
                       lag=10, 				      # Lag to detect the change point, DEFAULT: 5
                       duration_index=100, 	# Fitting duration after peak	, DEFAULT: 100
                       filename=filename, 	# filename for plotting			, DEFAULT: ""
                       num_peak=1)			    # The number of peak, 			, DEFAULT: 1
#%% --------------------- END FITTING -----------------------------------
fitted = output['fitted']
upper = output['upper_ci']
lower = output['lower_ci']
output_array = np.asarray( [t, fitted, lower, upper])
output_array = np.transpose(output_array)
# %% ------------------ WRITING RESULTs TO FILE -------------------------
np.savetxt(f'fitted_{f}'  , fitted, delimiter=",",  fmt='%f')
np.savetxt(f'upper_ci_{f}', upper, delimiter=",",  fmt='%f')
np.savetxt(f'lower_ci_{f}', lower, delimiter=",",  fmt='%f')
# %%
