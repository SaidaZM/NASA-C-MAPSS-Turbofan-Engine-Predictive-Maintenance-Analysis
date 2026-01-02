# ==========================================================
#      PROJET FMDS & MAINTENANCE PREDICTIVE - PYTHON VERSION
#      NASA C-MAPSS FD001  (Full Analysis)
#      Author : Saida Zmitri
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import weibull_min, gaussian_kde
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter.filedialog import askopenfilename

# ==========================================================
# 1. LOAD DATA (Automatic File Dialog)
# ==========================================================

print("\n🔍 Select your train_FD001.txt file…")

root = tk.Tk()
root.withdraw()
filepath = askopenfilename(title="Select train_FD001.txt", filetypes=[("Text Files","*.txt")])

if not filepath:
    raise Exception("❌ No file selected.")

print(f"📄 Loaded: {filepath}")

# Load file
df = pd.read_csv(filepath, sep=r"\s+", header=None)
df.dropna(axis=1, how="all", inplace=True)

colNames = ["id","cycle","setting1","setting2","setting3"] + \
           [f"s{i}" for i in range(1,22)]
df.columns = colNames[:df.shape[1]]

print("✅ Data loaded")

# ==========================================================
# 2. CLEAN MISSING DATA
# ==========================================================
df = df.interpolate(method="linear")

# ==========================================================
# 3. OUTLIER REMOVAL (3σ-rule)
# ==========================================================
sensors = df.columns[4:]

for col in sensors:
    mu = df[col].mean()
    sigma = df[col].std()
    outliers = abs(df[col] - mu) > 3 * sigma
    df.loc[outliers, col] = mu

# ==========================================================
# 4. NORMALIZATION (Min-Max)
# ==========================================================
scaler = MinMaxScaler()
df[sensors] = scaler.fit_transform(df[sensors])

# ==========================================================
# 5. RUL CALCULATION
# ==========================================================
max_cycle = df.groupby("id")["cycle"].max().rename("max_cycle")
df = df.merge(max_cycle, on="id", how="left")
df["RUL"] = df["max_cycle"] - df["cycle"]

# ==========================================================
# 6. PREPARE FEATURES
# ==========================================================
X = df[sensors].values
Y = df["RUL"].values

# ==========================================================
# 7. FMDS BASIC STATS
# ==========================================================
n_engines = df["id"].nunique()
MTBF = max_cycle.mean()
MTTR = 5
Disponibilite = MTBF / (MTBF + MTTR)

print(f"\n🔧 Number of engines: {n_engines}")
print(f"📌 MTBF: {MTBF:.2f} cycles")
print(f"📌 Availability: {Disponibilite*100:.2f}%")

# ==========================================================
# 8. SENSOR CORRELATION
# ==========================================================
plt.figure(figsize=(10,8))
sns.heatmap(df[sensors].corr(), cmap="coolwarm")
plt.title("Sensor Correlation Matrix")
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 9. LINEAR REGRESSION
# ==========================================================
lin = LinearRegression().fit(X, Y)
pred_lin = lin.predict(X)
rmse_lin = np.sqrt(np.mean((pred_lin - Y)**2))
print(f"\nRMSE Linear Regression: {rmse_lin:.2f}")

# ==========================================================
# 10. RANDOM FOREST
# ==========================================================
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5)
rf.fit(X, Y)
pred_rf = rf.predict(X)
rmse_rf = np.sqrt(np.mean((pred_rf - Y)**2))
print(f"RMSE Random Forest: {rmse_rf:.2f}")

# ==========================================================
# 11. RUL VISUALIZATION OF ENGINE 1
# ==========================================================
eng1 = df[df["id"]==1]
X1 = eng1[sensors]
y1 = eng1["RUL"]

plt.figure(figsize=(10,5))
plt.plot(eng1["cycle"], y1, label="True RUL")
plt.plot(eng1["cycle"], lin.predict(X1), label="Linear Pred")
plt.plot(eng1["cycle"], rf.predict(X1), label="RF Pred")
plt.legend()
plt.title("RUL prediction for engine 1")
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 12. SVM Classification (RUL < 30)
# ==========================================================
labels = (Y < 30).astype(int)
svm = SVC(kernel="rbf")
svm.fit(X, labels)
pred_svm = svm.predict(X)

cm = confusion_matrix(labels, pred_svm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("SVM: Healthy vs Near-Failure")
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 13. WEIBULL ESTIMATION (ROBUST + FIXED)
# ==========================================================
lifetimes = df.groupby('id')['max_cycle'].first().values

shape, loc, scale = weibull_min.fit(lifetimes, floc=0)
beta = shape
eta = scale

print(f"\n📈 Weibull parameters → beta={beta:.3f}, eta={eta:.3f}")

# Empirical survival
sorted_lifetimes = np.sort(lifetimes)
n = len(sorted_lifetimes)
ecdf_vals = np.arange(1, n+1) / n
R_emp = 1 - ecdf_vals

xx = np.linspace(0, sorted_lifetimes.max()*1.1, 300)

plt.figure()
plt.step(sorted_lifetimes, np.concatenate(([1.0], R_emp[:-1])),
         where='post', label="Empirical Survival")
plt.plot(xx, weibull_min.sf(xx, beta, scale=eta), "r--", label="Weibull Survival")
plt.title("Empirical Survival vs Weibull")
plt.legend(); plt.grid(True); 
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 14. KAPLAN–MEIER (NO CENSORING)
# ==========================================================
km = KaplanMeierFitter()
km.fit(lifetimes)

plt.figure()
km.plot_survival_function(label="KM")
plt.plot(xx, weibull_min.sf(xx, beta, scale=eta), 'r--', label="Weibull")
plt.title("KM vs Weibull")
plt.legend()
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 15. KM WITH 20% CENSORING
# ==========================================================
censor = np.zeros(n, dtype=int)
obs = lifetimes.copy()

idx = np.random.choice(n, int(0.2*n), replace=False)
for i in idx:
    obs[i] = np.random.uniform(1, lifetimes[i]*0.8)
    censor[i] = 1

km2 = KaplanMeierFitter()
km2.fit(obs, event_observed=1-censor)

plt.figure()
km.plot_survival_function(label="KM no censor")
km2.plot_survival_function(label="KM 20% censor")
plt.plot(xx, weibull_min.sf(xx, beta, scale=eta), 'r-.', label="Weibull")
plt.legend(); plt.title("Effect of Censoring")
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 16. NELSON–AALEN HAZARD
# ==========================================================
naf = NelsonAalenFitter()
naf.fit(obs, event_observed=1-censor)

plt.figure()
naf.plot_hazard(label="Nelson-Aalen Hazard", bandwidth=1.0)

plt.plot(xx, weibull_min.pdf(xx,beta,scale=eta)/weibull_min.sf(xx,beta,scale=eta),
         'r--',label="Weibull Hazard")
plt.legend(); plt.title("Hazard Comparison")
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 17. DEGRADATION SIMULATION
# ==========================================================
plt.figure(figsize=(10,6))
for i in range(3):
    T = lifetimes[i]
    t = np.arange(1,T+1)
    sig = t/T + 0.05*np.random.randn(T)
    sig[sig<0]=0
    plt.plot(t, sig, label=f"Engine {i+1}")

plt.title("Simulated Degradation Signals")
plt.legend()
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 18. CONFUSION MATRICES (BINARY + 3-CLASS)
# ==========================================================
tri_true = np.where(Y<=30,2, np.where(Y<=60,1,0))
tri_pred = np.where(pred_rf<=30,2, np.where(pred_rf<=60,1,0))

cm3 = confusion_matrix(tri_true, tri_pred)
disp3 = ConfusionMatrixDisplay(cm3)
disp3.plot()
plt.title("RF 3-Class Decision Matrix")
plt.ion()
plt.show(block=False)
plt.pause(0.8)


# ==========================================================
# 19. RUL STATISTICS
# ==========================================================
print("\n=== RUL Statistics ===")
print("Mean :", Y.mean())
print("Median:", np.median(Y))
print("Std   :", Y.std())
print("Percentiles:", np.percentile(Y,[10,25,50,75,90]))

plt.figure()
plt.hist(Y, bins=40, density=True, alpha=0.6)
kde2 = gaussian_kde(Y)
plt.plot(xx, kde2(xx), 'r-', label="KDE")
plt.legend(); plt.title("RUL Distribution")
plt.ion()
plt.show(block=False)
plt.pause(0.8)

input("Appuyez sur Entrée pour fermer toutes les figures...")

print("\n🎉 FMDS Full Python Analysis Completed Successfully!")

