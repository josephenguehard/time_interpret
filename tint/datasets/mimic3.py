import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import torch as th
import warnings

from datetime import timedelta
from getpass import getpass
from sklearn.impute import SimpleImputer

from tint.utils import get_progress_bars

from .dataset import DataModule

try:
    import psycopg2
except ImportError:
    psycopg2 = None


warnings.filterwarnings("ignore")

file_dir = os.path.dirname(__file__)

vital_IDs = [
    "HeartRate",
    "SysBP",
    "DiasBP",
    "MeanBP",
    "RespRate",
    "SpO2",
    "Glucose",
    "Temp",
]
lab_IDs = [
    "ANION GAP",
    "ALBUMIN",
    "BICARBONATE",
    "BILIRUBIN",
    "CREATININE",
    "CHLORIDE",
    "GLUCOSE",
    "HEMATOCRIT",
    "HEMOGLOBIN" "LACTATE",
    "MAGNESIUM",
    "PHOSPHATE",
    "PLATELET",
    "POTASSIUM",
    "PTT",
    "INR",
    "PT",
    "SODIUM",
    "BUN",
    "WBC",
]
eth_list = ["white", "black", "hispanic", "asian", "other"]

EPS = 1e-5


class Mimic3(DataModule):
    r"""
    MIMIC-III dataset.

    Download is set up according to this repository:
    https://github.com/sanatonek/time_series_explainability.

    .. warning::
        Using this dataset requires to have the MIMIC III data running on a
        local server. Please see https://mimic.mit.edu/docs/gettingstarted/local/install-mimic-locally-ubuntu/
        for more information.

    Args:
        data_dir (str): Where to download files.
        batch_size (int): Batch size. Default to 32
        n_folds (int): Number of folds for cross validation. If ``None``,
            the dataset is only split once between train and val using
            ``prop_val``. Default to ``None``
        fold (int): Index of the fold to use with cross-validation.
            Ignored if n_folds is None. Default to ``None``
        prop_val (float): Proportion of validation. Default to .2
        num_workers (int): Number of workers for the loaders. Default to 0
        seed (int): For the random split. Default to 42

    References:
        https://github.com/sanatonek/time_series_explainability/blob/master/data_generator/icu_mortality.py
        https://physionet.org/content/mimiciii/1.4/

    Examples:
        >>> from tint.datasets import Mimic3
        <BLANKLINE>
        >>> mimci3 = Mimic3()
        >>> mimci3.download(sqluser="your_username", split="train")
        >>> x_train = mimci3.preprocess(split="train")["x"]
        >>> y_train = mimci3.preprocess(split="train")["y"]
    """

    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "mimic3",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = None,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            n_folds=n_folds,
            fold=fold,
            num_workers=num_workers,
            seed=seed,
        )

        # Init mean and std
        self._mean = None
        self._std = None

    def download(
        self,
        sqluser: str = "mimicuser",
        prop_train: float = 0.8,
        split: str = "train",
    ):
        assert psycopg2 is not None, "You need to install psycopg2."

        random.seed(22891)

        sqlpass = getpass(prompt="sqlpass: ")

        # create a database connection and connect to local postgres
        # version of mimic
        dbname = "mimic"
        schema_name = "mimiciii"
        con = psycopg2.connect(
            dbname=dbname,
            user=sqluser,
            host="127.0.0.1",
            password=sqlpass,
        )
        cur = con.cursor()
        cur.execute("SET search_path to " + schema_name)

        # ========get the icu details

        # this query extracts the following:
        #   Unique ids for the admission, patient and icu stay
        #   Patient gender
        #   diagnosis
        #   age
        #   ethnicity
        #   admission type
        #   first hospital stay
        #   first icu stay?
        #   mortality within a week

        denquery = """
            --ie is the icustays table 
            --adm is the admissions table 
            SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
            , pat.gender
            , adm.admittime, adm.dischtime, adm.diagnosis
            , ROUND( (CAST(adm.dischtime AS DATE) - CAST(adm.admittime AS DATE)) , 4) AS los_hospital
            , ROUND( (CAST(adm.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365, 4) AS age
            , adm.ethnicity, adm.ADMISSION_TYPE
            --, adm.hospital_expire_flag
            , CASE when adm.deathtime between ie.intime and ie.outtime THEN 1 ELSE 0 END AS mort_icu
            , DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
            , CASE
                WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
                ELSE 0 END AS first_hosp_stay
            -- icu level factors
            , ie.intime, ie.outtime
            , ie.FIRST_CAREUNIT
            , ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
            , DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq
            , CASE
                WHEN adm.deathtime between ie.intime and ie.intime + interval '168' hour THEN 1 ELSE 0 END AS mort_week
            -- first ICU stay *for the current hospitalization*
            , CASE
                WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
                ELSE 0 END AS first_icu_stay
            FROM icustays ie
            INNER JOIN admissions adm
                ON ie.hadm_id = adm.hadm_id
            INNER JOIN patients pat
                ON ie.subject_id = pat.subject_id
            WHERE adm.has_chartevents_data = 1
            ORDER BY ie.subject_id, adm.admittime, ie.intime;
            """

        den = pd.read_sql_query(denquery, con)

        # drop patients with less than 48 hour
        den["los_icu_hr"] = (den.outtime - den.intime).astype("timedelta64[h]")
        den = den[(den.los_icu_hr >= 48)]
        den = den[(den.age < 300)]
        den.drop("los_icu_hr", 1, inplace=True)

        # clean up
        den["adult_icu"] = np.where(
            den["first_careunit"].isin(["PICU", "NICU"]), 0, 1
        )
        den["gender"] = np.where(den["gender"] == "M", 1, 0)
        den.ethnicity = den.ethnicity.str.lower()
        den.ethnicity.loc[(den.ethnicity.str.contains("^white"))] = "white"
        den.ethnicity.loc[(den.ethnicity.str.contains("^black"))] = "black"
        den.ethnicity.loc[
            (den.ethnicity.str.contains("^hisp"))
            | (den.ethnicity.str.contains("^latin"))
        ] = "hispanic"
        den.ethnicity.loc[(den.ethnicity.str.contains("^asia"))] = "asian"
        den.ethnicity.loc[
            ~(
                den.ethnicity.str.contains(
                    "|".join(["white", "black", "hispanic", "asian"])
                )
            )
        ] = "other"

        den.drop(
            [
                "hospstay_seq",
                "los_icu",
                "icustay_seq",
                "admittime",
                "dischtime",
                "los_hospital",
                "intime",
                "outtime",
                "first_careunit",
            ],
            1,
            inplace=True,
        )

        # ========= 48 hour vitals query
        # these are the normal ranges. useful to clean up the data

        vitquery = """
            -- This query pivots the vital signs for the first 48 hours of a patient's stay
            -- Vital signs include heart rate, blood pressure, respiration rate, and temperature
            -- DROP MATERIALIZED VIEW IF EXISTS vitalsfirstday CASCADE;
            -- create materialized view vitalsfirstday as
            SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.VitalID, pvt.VitalValue, pvt.VitalChartTime
            FROM  (
              select ie.subject_id, ie.hadm_id, ie.icustay_id, ce.charttime as VitalChartTime 
              , case
                when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then 'HeartRate'
                when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then 'SysBP'
                when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then 'DiasBP'
                when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then 'MeanBP'
                when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then 'RespRate'
                when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then 'Temp' -- converted to degC in valuenum call
                when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then 'Temp'
                when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then 'SpO2'
                when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then 'Glucose'
                else null end as VitalID
                , case
                when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then valuenum -- HeartRate
                when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then valuenum -- SysBP
                when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then valuenum -- DiasBP
                when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then valuenum -- MeanBP
                when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then valuenum -- RespRate
                when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then (valuenum-32)/1.8 -- TempF, convert to degC
                when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then valuenum -- TempC
                when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then valuenum -- SpO2
                when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then valuenum -- Glucose
                else null end as VitalValue
              from icustays ie
              left join chartevents ce
              on ie.subject_id = ce.subject_id and ie.hadm_id = ce.hadm_id and ie.icustay_id = ce.icustay_id
              and ce.charttime between ie.intime and ie.intime + interval '48' hour
              -- exclude rows marked as error
              and ce.error IS DISTINCT FROM 1 
              where ce.itemid in
              (
              -- HEART RATE
              211, --"Heart Rate"
              220045, --"Heart Rate"
              -- Systolic/diastolic
              51, --	Arterial BP [Systolic]
              442, --	Manual BP [Systolic]
              455, --	NBP [Systolic]
              6701, --	Arterial BP #2 [Systolic]
              220179, --	Non Invasive Blood Pressure systolic
              220050, --	Arterial Blood Pressure systolic
              8368, --	Arterial BP [Diastolic]
              8440, --	Manual BP [Diastolic]
              8441, --	NBP [Diastolic]
              8555, --	Arterial BP #2 [Diastolic]
              220180, --	Non Invasive Blood Pressure diastolic
              220051, --	Arterial Blood Pressure diastolic
              -- MEAN ARTERIAL PRESSURE
              456, --"NBP Mean"
              52, --"Arterial BP Mean"
              6702, --	Arterial BP Mean #2
              443, --	Manual BP Mean(calc)
              220052, --"Arterial Blood Pressure mean"
              220181, --"Non Invasive Blood Pressure mean"
              225312, --"ART BP mean"
              -- RESPIRATORY RATE
              618,--	Respiratory Rate
              615,--	Resp Rate (Total)
              220210,--	Respiratory Rate
              224690, --	Respiratory Rate (Total)
              -- SPO2, peripheral
              646, 220277,
              -- GLUCOSE, both lab and fingerstick
              807,--	Fingerstick Glucose
              811,--	Glucose (70-105)
              1529,--	Glucose
              3745,--	BloodGlucose
              3744,--	Blood Glucose
              225664,--	Glucose finger stick
              220621,--	Glucose (serum)
              226537,--	Glucose (whole blood)
              -- TEMPERATURE
              223762, -- "Temperature Celsius"
              676,	-- "Temperature C"
              223761, -- "Temperature Fahrenheit"
              678 --	"Temperature F"
              ) 
            ) pvt
            where VitalID is not null
            order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.VitalID, pvt.VitalChartTime;
            """
        vit48 = pd.read_sql_query(vitquery, con)
        vit48.isnull().sum()

        # ===============48 hour labs query
        # This query extracts the lab events in the first 48 hours
        labquery = """
            WITH pvt AS (
              --- ie is the icu stay 
              --- ad is the admissions table 
              --- le is the lab events table 
              SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime as LabChartTime
              , CASE
                    when le.itemid = 50868 then 'ANION GAP'
                    when le.itemid = 50862 then 'ALBUMIN'
                    when le.itemid = 50882 then 'BICARBONATE'
                    when le.itemid = 50885 then 'BILIRUBIN'
                    when le.itemid = 50912 then 'CREATININE'
                    when le.itemid = 50806 then 'CHLORIDE'
                    when le.itemid = 50902 then 'CHLORIDE'
                    when le.itemid = 50809 then 'GLUCOSE'
                    when le.itemid = 50931 then 'GLUCOSE'
                    when le.itemid = 50810 then 'HEMATOCRIT'
                    when le.itemid = 51221 then 'HEMATOCRIT'
                    when le.itemid = 50811 then 'HEMOGLOBIN'
                    when le.itemid = 51222 then 'HEMOGLOBIN'
                    when le.itemid = 50813 then 'LACTATE'
                    when le.itemid = 50960 then 'MAGNESIUM'
                    when le.itemid = 50970 then 'PHOSPHATE'
                    when le.itemid = 51265 then 'PLATELET'
                    when le.itemid = 50822 then 'POTASSIUM'
                    when le.itemid = 50971 then 'POTASSIUM'
                    when le.itemid = 51275 then 'PTT'
                    when le.itemid = 51237 then 'INR'
                    when le.itemid = 51274 then 'PT'
                    when le.itemid = 50824 then 'SODIUM'
                    when le.itemid = 50983 then 'SODIUM'
                    when le.itemid = 51006 then 'BUN'
                    when le.itemid = 51300 then 'WBC'
                    when le.itemid = 51301 then 'WBC'
                  ELSE null
                  END AS label
  
              , -- add in some sanity checks on the values
                CASE
                  when le.itemid = 50862 and le.valuenum >    10 then null -- g/dL 'ALBUMIN'
                  when le.itemid = 50868 and le.valuenum > 10000 then null -- mEq/L 'ANION GAP'
                  when le.itemid = 50882 and le.valuenum > 10000 then null -- mEq/L 'BICARBONATE'
                  when le.itemid = 50885 and le.valuenum >   150 then null -- mg/dL 'BILIRUBIN'
                  when le.itemid = 50806 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
                  when le.itemid = 50902 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
                  when le.itemid = 50912 and le.valuenum >   150 then null -- mg/dL 'CREATININE'
                  when le.itemid = 50809 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
                  when le.itemid = 50931 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
                  when le.itemid = 50810 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
                  when le.itemid = 51221 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
                  when le.itemid = 50811 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
                  when le.itemid = 51222 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
                  when le.itemid = 50813 and le.valuenum >    50 then null -- mmol/L 'LACTATE'
                  when le.itemid = 50960 and le.valuenum >    60 then null -- mmol/L 'MAGNESIUM'
                  when le.itemid = 50970 and le.valuenum >    60 then null -- mg/dL 'PHOSPHATE'
                  when le.itemid = 51265 and le.valuenum > 10000 then null -- K/uL 'PLATELET'
                  when le.itemid = 50822 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
                  when le.itemid = 50971 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
                  when le.itemid = 51275 and le.valuenum >   150 then null -- sec 'PTT'
                  when le.itemid = 51237 and le.valuenum >    50 then null -- 'INR'
                  when le.itemid = 51274 and le.valuenum >   150 then null -- sec 'PT'
                  when le.itemid = 50824 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
                  when le.itemid = 50983 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
                  when le.itemid = 51006 and le.valuenum >   300 then null -- 'BUN'
                  when le.itemid = 51300 and le.valuenum >  1000 then null -- 'WBC'
                  when le.itemid = 51301 and le.valuenum >  1000 then null -- 'WBC'
                ELSE le.valuenum
                END AS LabValue
              FROM icustays ie
              LEFT JOIN labevents le
                ON le.subject_id = ie.subject_id 
                AND le.hadm_id = ie.hadm_id
                AND le.charttime between (ie.intime) AND (ie.intime + interval '48' hour)
                AND le.itemid IN
                (
                  -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
                  50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
                  50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
                  50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
                  50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
                  50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
                  50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
                  50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
                  50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
                  50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
                  51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
                  50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
                  51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
                  50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
                  50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
                  50960, -- MAGNESIUM | CHEMISTRY | BLOOD | 664191
                  50970, -- PHOSPHATE | CHEMISTRY | BLOOD | 590524
                  51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
                  50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
                  50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
                  51275, -- PTT | HEMATOLOGY | BLOOD | 474937
                  51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
                  51274, -- PT | HEMATOLOGY | BLOOD | 469090
                  50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
                  50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
                  51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
                  51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
                  51300  -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
                )
                AND le.valuenum IS NOT null 
                AND le.valuenum > 0 -- lab values cannot be 0 and cannot be negative
  
                LEFT JOIN admissions ad
                ON ie.subject_id = ad.subject_id
                AND ie.hadm_id = ad.hadm_id
  
            )
            SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.LabChartTime, pvt.label, pvt.LabValue 
            From pvt
            where pvt.label is not NULL
            ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.label, pvt.LabChartTime;
            """

        lab48 = pd.read_sql_query(labquery, con)

        # =====combine all variables
        mort_vital = den.merge(
            vit48, how="left", on=["subject_id", "hadm_id", "icustay_id"]
        )
        mort_lab = den.merge(
            lab48, how="left", on=["subject_id", "hadm_id", "icustay_id"]
        )

        # create means by age group and gender
        mort_vital["age_group"] = pd.cut(
            mort_vital["age"],
            [-1, 5, 10, 15, 20, 25, 40, 60, 80, 200],
            labels=[
                "l5",
                "5_10",
                "10_15",
                "15_20",
                "20_25",
                "25_40",
                "40_60",
                "60_80",
                "80p",
            ],
        )
        mort_lab["age_group"] = pd.cut(
            mort_lab["age"],
            [-1, 5, 10, 15, 20, 25, 40, 60, 80, 200],
            labels=[
                "l5",
                "5_10",
                "10_15",
                "15_20",
                "20_25",
                "25_40",
                "40_60",
                "60_80",
                "80p",
            ],
        )

        # one missing variable
        adult_vital = mort_vital[(mort_vital.adult_icu == 1)]
        adult_lab = mort_lab[(mort_lab.adult_icu == 1)]
        adult_vital.drop(columns=["adult_icu"], inplace=True)
        adult_lab.drop(columns=["adult_icu"], inplace=True)

        # Save files
        adult_vital.to_csv(
            os.path.join(self.data_dir, "adult_icu_vital.gz"),
            compression="gzip",
            index=False,
        )
        mort_lab.to_csv(
            os.path.join(self.data_dir, "adult_icu_lab.gz"),
            compression="gzip",
            index=False,
        )

        # Drop NAs
        adult_vital = adult_vital.dropna(subset=["vitalid"])
        mort_lab = mort_lab.dropna(subset=["label"])

        # Get unique ids
        icu_ids = adult_vital.icustay_id.unique()

        # Create arrays
        x = np.zeros((len(icu_ids), 12, 48))
        x_lab = np.zeros((len(icu_ids), len(lab_IDs), 48))
        x_impute = np.zeros((len(icu_ids), 12, 48))
        y = np.zeros((len(icu_ids),))
        imp_mean = SimpleImputer(strategy="mean")

        missing_ids = []
        missing_map = np.zeros((len(icu_ids), 12))
        missing_map_lab = np.zeros((len(icu_ids), len(lab_IDs)))

        nan_map = np.zeros((len(icu_ids), len(lab_IDs) + 12))

        # Create ethnicity encoding

        # Populate data
        pbar = get_progress_bars()(enumerate(icu_ids), total=len(icu_ids))
        for i, icu_id in pbar:
            patient_data = adult_vital.loc[adult_vital["icustay_id"] == icu_id]
            patient_data["vitalcharttime"] = patient_data[
                "vitalcharttime"
            ].astype("datetime64[s]")
            patient_lab_data = mort_lab.loc[mort_lab["icustay_id"] == icu_id]
            patient_lab_data["labcharttime"] = patient_lab_data[
                "labcharttime"
            ].astype("datetime64[s]")

            admit_time = patient_data["vitalcharttime"].min()
            n_missing_vitals = 0

            # Extract demographics and repeat them over time
            x[i, -4, :] = int(patient_data["gender"].iloc[0])
            x[i, -3, :] = int(patient_data["age"].iloc[0])
            x[i, -2, :] = ethnicity_encoder(
                patient_data["ethnicity"].iloc[0], patient_data
            )
            x[i, -1, :] = int(patient_data["first_icu_stay"].iloc[0])
            y[i] = int(patient_data["mort_icu"].iloc[0])

            # Extract vital measurement information
            vitals = patient_data.vitalid.unique()
            for vital in vitals:
                try:
                    vital_IDs.index(vital)
                    signal = patient_data[patient_data["vitalid"] == vital]
                    quantized_signal, _ = quantize_signal(
                        signal,
                        start=admit_time,
                        step_size=1,
                        n_steps=48,
                        value_column="vitalvalue",
                        charttime_column="vitalcharttime",
                    )
                    nan_arr, nan_count = check_nan(quantized_signal)
                    x[i, vital_IDs.index(vital), :] = np.array(
                        quantized_signal
                    )
                    nan_map[
                        i, len(lab_IDs) + vital_IDs.index(vital)
                    ] = nan_count
                    if nan_count == 48:
                        n_missing_vitals = +1
                        missing_map[i, vital_IDs.index(vital)] = 1
                    else:
                        x_impute[i, :, :] = imp_mean.fit_transform(
                            x[i, :, :].T
                        ).T
                except:  # noqa: E722
                    pass

            # Extract lab measurement informations
            labs = patient_lab_data.label.unique()
            for lab in labs:
                try:
                    lab_IDs.index(lab)
                    lab_measures = patient_lab_data[
                        patient_lab_data["label"] == lab
                    ]
                    quantized_lab, quantized_measures = quantize_signal(
                        lab_measures,
                        start=admit_time,
                        step_size=1,
                        n_steps=48,
                        value_column="labvalue",
                        charttime_column="labcharttime",
                    )
                    nan_arr, nan_count = check_nan(quantized_lab)
                    x_lab[i, lab_IDs.index(lab), :] = np.array(quantized_lab)
                    nan_map[i, lab_IDs.index(lab)] = nan_count
                    if nan_count == 48:
                        missing_map_lab[i, lab_IDs.index(lab)] = 1
                except:  # noqa: E722
                    pass

            # Remove a patient that is missing a measurement for the entire 48 hours
            if n_missing_vitals > 0:
                missing_ids.append(i)

        # Record statistics of the dataset, remove missing samples and save the signals
        f = open(os.path.join(self.data_dir, "stats.txt"), "a")
        f.write(
            "\n ******************* Before removing missing *********************"
        )
        f.write(
            "\n Number of patients: "
            + str(len(y))
            + "\n Number of patients who died within their stay: "
            + str(np.count_nonzero(y))
        )
        f.write("\nMissingness report for Vital signals")
        for i, vital in enumerate(vital_IDs):
            f.write(
                "\nMissingness for %s: %.2f"
                % (vital, np.count_nonzero(missing_map[:, i]) / len(icu_ids))
            )
            f.write("\n")
        f.write("\nMissingness report for Vital signals")
        for i, lab in enumerate(lab_IDs):
            f.write(
                "\nMissingness for %s: %.2f"
                % (lab, np.count_nonzero(missing_map_lab[:, i]) / len(icu_ids))
            )
            f.write("\n")

        x_lab = np.delete(x_lab, missing_ids, axis=0)
        x_impute = np.delete(x_impute, missing_ids, axis=0)
        y = np.delete(y, missing_ids, axis=0)
        nan_map = np.delete(nan_map, missing_ids, axis=0)

        x_lab_impute = impute_lab(x_lab)
        missing_map = np.delete(missing_map, missing_ids, axis=0)
        missing_map_lab = np.delete(missing_map_lab, missing_ids, axis=0)
        all_data = np.concatenate((x_lab_impute, x_impute), axis=1)
        f.write(
            "\n ******************* After removing missing *********************"
        )
        f.write(
            "\n Final number of patients: "
            + str(len(y))
            + "\n Number of patients who died within their stay: "
            + str(np.count_nonzero(y))
        )
        f.write("\nMissingness report for Vital signals")
        for i, vital in enumerate(vital_IDs):
            f.write(
                "\nMissingness for %s: %.2f"
                % (vital, np.count_nonzero(missing_map[:, i]) / len(icu_ids))
            )
            f.write("\n")
        f.write("\nMissingness report for Vital signals")
        for i, lab in enumerate(lab_IDs):
            f.write(
                "\nMissingness for %s: %.2f"
                % (lab, np.count_nonzero(missing_map_lab[:, i]) / len(icu_ids))
            )
            f.write("\n")
        f.close()

        samples = [
            (all_data[i, :, :], y[i], nan_map[i, :]) for i in range(len(y))
        ]

        # Split train and test
        train_size = int(len(samples) * prop_train)
        train_samples = samples[:train_size]
        test_samples = samples[train_size:]

        # Save preprocessed data
        with open(
            os.path.join(
                self.data_dir, "train_patient_vital_preprocessed.pkl"
            ),
            "wb",
        ) as f:
            pkl.dump(train_samples, f)
        with open(
            os.path.join(self.data_dir, "test_patient_vital_preprocessed.pkl"),
            "wb",
        ) as f:
            pkl.dump(test_samples, f)

    def prepare_data(self):
        if not os.path.exists(
            os.path.join(self.data_dir, "train_patient_vital_preprocessed.pkl")
        ) or not os.path.join(
            self.data_dir, "test_patient_vital_preprocessed.pkl"
        ):
            sqluser = input("sqluser: ")
            self.download(sqluser=sqluser)

    def preprocess(self, split: str = "train") -> dict:
        # Load data
        file = os.path.join(self.data_dir, f"{split}_")
        with open(file + "patient_vital_preprocessed.pkl", "rb") as fp:
            data = pkl.load(fp)

        features = th.Tensor([x for (x, y, z) in data]).transpose(1, 2)
        labels = th.Tensor([y for (x, y, z) in data])

        # Compute mean and std
        if split == "train":
            self._mean = features.reshape(-1, features.shape[-1]).mean(0)
            self._std = features.reshape(-1, features.shape[-1]).mean(0)
        else:
            assert split == "test", "split must be train or test"

        assert (
            self._mean is not None
        ), "You must call preprocess('train') first"

        # Normalise
        mean = self._mean.unsqueeze(0).unsqueeze(0)
        std = self._std.unsqueeze(0).unsqueeze(0)
        features = (features - mean) / (std + EPS)

        return {
            "x": features.float(),
            "y": labels.long(),
        }


def quantize_signal(
    signal, start, step_size, n_steps, value_column, charttime_column
):
    quantized_signal = []
    quantized_counts = np.zeros((n_steps,))
    s = start
    u = start + timedelta(hours=step_size)
    for i in range(n_steps):
        signal_window = signal[value_column][
            (signal[charttime_column] > s) & (signal[charttime_column] < u)
        ]
        quantized_signal.append(signal_window.mean())
        quantized_counts[i] = len(signal_window)
        s = u
        u = s + timedelta(hours=step_size)
    return quantized_signal, quantized_counts


def check_nan(a):
    a = np.array(a)
    nan_arr = np.isnan(a).astype(int)
    nan_count = np.count_nonzero(nan_arr)
    return nan_arr, nan_count


def forward_impute(x, nan_arr):
    x_impute = x.copy()
    first_value = 0
    while first_value < len(x) and nan_arr[first_value] == 1:
        first_value += 1
    last = x_impute[first_value]
    for i, measurement in enumerate(x):
        if nan_arr[i] == 1:
            x_impute[i] = last
        else:
            last = measurement
    return x_impute


def impute_lab(lab_data):
    imputer = SimpleImputer(strategy="mean")
    lab_data_impute = lab_data.copy()
    imputer.fit(lab_data.reshape((-1, lab_data.shape[1])))
    for i, patient in enumerate(lab_data):
        for j, signal in enumerate(patient):
            nan_arr, nan_count = check_nan(signal)
            if nan_count != len(signal):
                lab_data_impute[i, j, :] = forward_impute(signal, nan_arr)
    lab_data_impute = np.array(
        [imputer.transform(sample.T).T for sample in lab_data_impute]
    )
    return lab_data_impute


def ethnicity_encoder(eth, patient_data):
    return (
        0
        if eth == "0"
        else eth_list.index(patient_data["ethnicity"].iloc[0]) + 1
    )
