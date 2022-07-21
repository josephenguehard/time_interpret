import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import torch as th

from torch.nn.functional import one_hot as one_hot_
from torch.nn.utils.rnn import pad_sequence
from typing import List, Union

from tint.utils import get_progress_bars
from .dataset import DataModule
from .utils import create_labels, Fasttext

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None

try:
    from pandarallel import pandarallel
except ImportError:
    pandarallel = None


tqdm = get_progress_bars()
file_dir = os.path.dirname(__file__)


class BioBank(DataModule):
    """
    BioBank dataset.

    Args:
        discretised (bool): Whether to return a discretised dataset or not.
            Default to ``True``
        granularity (str, int): The time granularity.
            Default to a year: 24*365.25 hours
        maximum_time (int): Maximum time to record. Default to 1,000,000 hours
            which is around 114 years
        one_hot (bool): Whether to one hot encode the read codes or not.
            Default to ``True``
        fasttext (Fasttext): A Fasttext model to encode categorical features.
            Default to ``None``
        time_to_task (float): Special arg for diabetes task. Stops the
            recording before diabetes happens. Default to .5
        std_time_to_task (float): Add randomness into when to stop recording.
            Default to .2
        data_dir (str): Where to download files.
        batch_size (int): Batch size. Default to 32
        prop_val (float): Proportion of validation. Default to .2
        num_workers (int): Number of workers for the loaders. Default to 0
        seed (int): For the random split. Default to 42

    References:
        https://www.ukbiobank.ac.uk
    """

    def __init__(
        self,
        label: str = None,
        discretised: bool = False,
        granularity: int = 1,
        maximum_time: int = 115,
        one_hot: bool = False,
        fasttext: Fasttext = None,
        time_to_task: float = 0.5,
        std_time_to_task: float = 0.2,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "biobank",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            num_workers=num_workers,
            seed=seed,
        )

        self.discretised = discretised
        self.granularity = granularity
        self.maximum_time = maximum_time
        self.one_hot = one_hot
        self.fasttext = fasttext
        self.time_to_task = time_to_task
        self.std_time_to_task = std_time_to_task

        with open(os.path.join(file_dir, "utils", "read_3_2.json"), "r") as fp:
            self.read_3_2 = json.load(fp=fp)
        with open(os.path.join(file_dir, "utils", "labels.json"), "r") as fp:
            self.labels = json.load(fp=fp)

        # We drop term codes for all labels
        self.labels = {k: [x[:5] for x in v] for k, v in self.labels.items()}

        # Set the label or use the default type 2 diabetes
        self.label = label or self.labels["Type II diabetes mellitus (4)"]

    def download(
        self,
        split: str = "train",
        verbose: Union[int, bool] = False,
    ):
        # Set tqdm if necessary
        if verbose:
            assert tqdm is not None, "tqdm must be installed."
        pbar = tqdm(range(4), leave=True) if verbose else None
        pbar.set_description("Load Metadata") if verbose else None

        # Init pandarallel
        cpu_count = mp.cpu_count()
        assert pandarallel is not None, "pandarallel is not installed."
        pandarallel.initialize(
            nb_workers=max(1, cpu_count - 1),
            progress_bar=max(0, verbose - 1),
            verbose=0,
            use_memory_fs=False,
        )

        # Query Metadata
        assert bigquery is not None, "google-cloud-bigquery must be installed."
        client = bigquery.Client()

        query = f"""
                SELECT *
                FROM `dsap-prod-uk-biobank-ml.bio_bank.ukb_core` as ukb
                """
        metadata = client.query(query=query).to_dataframe()
        metadata = metadata.dropna(axis=1, thresh=1)

        # Convert eventual datetime object to datetime
        columns = metadata.select_dtypes(include="object").columns
        for column in columns:
            try:
                metadata[column] = metadata[column].astype("float")
            except:
                try:
                    metadata[column] = pd.to_datetime(metadata[column])
                except TypeError:
                    continue
                except ValueError:
                    continue

        # Convert dates in metadata to years
        columns = metadata.select_dtypes(include="datetime").columns
        columns = list(set(columns) - {"_34_0_0"})
        for column in columns:
            metadata[column] = metadata[[column, "_34_0_0"]].parallel_apply(
                lambda x: np.nan
                if pd.isna(x[0]) or pd.isna(x[1])
                else 1970 + x[0].timestamp() / 3600 / 24 / 365.25 - x[1],
                axis=1,
            )

        # Update dob
        metadata["_34_0_0"] = metadata["_34_0_0"].parallel_apply(
            lambda x: np.nan if pd.isna(x) else x
        )

        # Drop non-recognised columns
        metadata = metadata.select_dtypes(exclude="object")

        # Update tqdm
        pbar.update() if verbose else None
        pbar.set_description("Load GP data") if verbose else None

        # Query GP data
        query = f"""
                SELECT *
                FROM `dsap-prod-uk-biobank-ml.bio_bank.gp_clinical`
                """
        df = client.query(query=query).to_dataframe()

        # Filter df
        df = df[
            df.read_3.isin(list(self.read_3_2.keys())) + ~df.read_2.isnull()
        ]
        df = df[df["event_dt"].notnull()]

        # Convert read 3 codes to read 2
        df["read"] = df[["read_2", "read_3"]].parallel_apply(
            lambda x: x[0] if x[0] is not None else self.read_3_2[x[1]], axis=1
        )

        # Remove None types and unknown values
        df = df[df["read"].notnull()]
        df.read = df.read.parallel_apply(lambda x: x[:5] if len(x) > 5 else x)

        # Convert dates
        df.event_dt = pd.to_datetime(df.event_dt).parallel_apply(
            lambda x: x.timestamp()
        )
        df.event_dt = df.event_dt.parallel_apply(
            lambda x: x if pd.isna(x) else x / 3600 / 24 / 365.25
        )

        # Sort by timestamp
        df = df.sort_values(by="event_dt")

        # Add year of birth to df and subtract it, remove negative values
        df = pd.merge(df, metadata[["eid", "_34_0_0"]], how="inner", on="eid")
        df["event_dt"] = df[["event_dt", "_34_0_0"]].parallel_apply(
            lambda x: np.nan
            if pd.isna(x[0]) or pd.isna(x[1])
            else 1970 + x[0] - x[1],
            axis=1,
        )
        df = df[df["event_dt"] >= 0]

        # Create codes_to_idx data
        unique_codes = df.read.unique()
        codes_to_idx = {k: i + 1 for i, k in enumerate(unique_codes)}

        # Update tqdm
        pbar.update() if verbose else None
        pbar.set_description("Group patients") if verbose else None

        # Group per patient
        if verbose:
            df = (
                df[["eid", "event_dt", "read"]]
                .groupby(["eid", "event_dt"])
                .progress_aggregate(list)
                .reset_index()
            )
        else:
            df = (
                df[["eid", "event_dt", "read"]]
                .groupby(["eid", "event_dt"])
                .agg(list)
                .reset_index()
            )
        read = df.groupby("eid").read.apply(list).reset_index(name="read")
        times = (
            df.groupby("eid").event_dt.apply(list).reset_index(name="times")
        )

        # Merge with metadata and save dataframe
        df = pd.merge(read, times, how="inner", on="eid")
        df = pd.merge(df, metadata, how="inner", on="eid")
        df.to_csv(os.path.join(self.data_dir, "biobank_data.csv"), index=False)

        # Save codes_to_idx
        with open(os.path.join(self.data_dir, "codes_to_idx.json"), "w") as fp:
            json.dump(obj=codes_to_idx, fp=fp)

        # Save text file for fasttext training
        with open(os.path.join(self.data_dir, "codes_text.txt"), "w") as fp:
            events = df.read
            for patient in events:
                for record in patient:
                    for label in record:
                        label = codes_to_idx.get(label, None)
                        if label is not None:
                            fp.write(str(label) + " ")
                fp.write("\n")

        # Update tqdm
        pbar.update() if verbose else None

    def preprocess(
        self,
        split: str = "train",
        verbose: Union[bool, int] = False,
    ) -> (th.Tensor, th.Tensor):
        # Init pandarallel
        cpu_count = mp.cpu_count()
        assert pandarallel is not None, "pandarallel is not installed."
        pandarallel.initialize(
            nb_workers=max(1, cpu_count - 1),
            progress_bar=max(0, verbose - 1),
            verbose=0,
            use_memory_fs=False,
        )

        # Load data
        df = pd.read_csv(os.path.join(self.data_dir, "biobank_data.csv"))
        df.times = df.times.parallel_apply(eval)
        df.read = df.read.parallel_apply(eval)

        # Load codes_to_idx
        with open(os.path.join(self.data_dir, "codes_to_idx.json"), "r") as fp:
            codes_to_idx = json.load(fp=fp)

        # Extract times and metadata from dataframe
        times = df.times.values
        times = [th.Tensor(x).type(th.float32) for x in times]

        metadata = df.drop(["eid", "times", "read"], axis=1).values
        metadata = [th.Tensor(x).type(th.float32) for x in metadata]

        # Replace codes with int
        events = df.read.apply(
            lambda x: [[codes_to_idx[z] for z in y] for y in x]
        )

        if self.discretised:
            labels, mask = self.build_discretised_labels(
                events=events,
                times=times,
            )
            events, metadata, times = self.build_discretised_features(
                events=events,
                times=times,
                metadata=metadata,
                codes_to_idx=codes_to_idx,
                verbose=verbose,
            )
            events[mask.bool()] = 0.

        else:
            labels, mask = self.build_labels(
                events=events,
                times=times,
            )
            events, metadata, times = self.build_features(
                events=events,
                times=times,
                metadata=metadata,
                codes_to_idx=codes_to_idx,
                verbose=verbose,
            )
            events = [x[y.bool()] for x, y in zip(events, mask)]
            times = [x[y.bool()] for x, y in zip(times, mask)]

    def build_features(
        self,
        events: List[th.Tensor],
        times: List[th.Tensor],
        metadata: List[th.Tensor],
        codes_to_idx: dict,
        verbose: Union[bool, int] = False,
    ):
        """
        Build features.

        Args:
            events (list): The read codes.
            times (list): Times of each event.
            metadata (list): List of metadata.
            codes_to_idx (dict): Dictionary linking codes to indexes.
            verbose (bool, int): Verbosity level. Default to ``False``

        Returns:
            Preprocessed features.
        """
        if verbose:
            events = tqdm(events, total=len(events), leave=False)

        if self.one_hot and self.fasttext is None:
            events = [
                th.stack(
                    [
                        one_hot_(
                            th.Tensor(x).long(),
                            num_classes=len(codes_to_idx),
                        )
                        .sum(0)
                        .type(th.float32)
                        for x in r
                    ]
                )
                for r in events
            ]  # [D,L,M]
        elif self.fasttext is not None:
            events = [
                th.stack(
                    [
                        self.fasttext.transform(th.Tensor(x).long())
                            .sum(0)
                            .type(th.float32)
                        for x in r
                    ]
                )
                for r in events
            ]  # [D,L,M]
        else:
            max_sim_events = max([max([len(x) for x in y]) for y in events])
            events_ = [
                pad_sequence([th.Tensor(x) for x in y], batch_first=True)
                for y in events
            ]
            events = [th.zeros((x.shape[0], max_sim_events)) for x in events_]
            for x, y in zip(events, events_):
                x[:, : y.shape[1]] = y

        return events, times, metadata

    def build_discretised_features(
        self,
        events: List[th.Tensor],
        times: List[th.Tensor],
        metadata: List[th.Tensor],
        codes_to_idx: dict,
        verbose: Union[bool, int] = False,
    ):
        """
        Build discretised features.

        Args:
            events (list): The read codes.
            times (list): Times of each event.
            metadata (list): List of metadata.
            codes_to_idx (dict): Dictionary linking codes to indexes.
            verbose (bool, int): Verbosity level. Default to ``False``

        Returns:
            Preprocessed features.
        """
        if verbose:
            events = tqdm(events, total=len(events), leave=False)

        if self.one_hot and self.fasttext is None:
            codes = th.zeros(
                (
                    len(events),
                    int(self.maximum_time / self.granularity) + 1,
                    len(codes_to_idx),
                )
            ).type(th.float32)
            for i, r in enumerate(events):
                f = th.stack(
                    [
                        one_hot_(
                            th.Tensor(x).long(),
                            num_classes=len(codes_to_idx),
                        )
                        .sum(0)
                        .type(th.float32)
                        for x in r
                    ]
                )
                codes[i].index_add_(0, times[i], f)
        elif self.fasttext is not None:
            codes = th.zeros(
                (
                    len(events),
                    int(self.maximum_time / self.granularity) + 1,
                    self.fasttext.emb_dim,
                )
            ).type(th.float32)
            for i, r in enumerate(events):
                f = th.stack(
                    [
                        self.fasttext.transform(th.Tensor(x).long())
                        .sum(0)
                        .type(th.float32)
                        for x in r
                    ]
                )
                codes[i].index_add_(0, times[i], f)
        else:
            raise NotImplementedError(
                "When using discretised data, "
                "categorical features must be encoded."
            )

        return codes, metadata, None

    def build_labels(
            self,
            events: list,
            times: list,
    ) -> (list, list):
        """
        Build labels.

        Args:
            events (list): Dict of events.
            times (list): List of times.

        Returns:
            (list, list): Two lists of labels and tasks
        """
        labels, end_of_records = create_labels(
            events=events,
            event_times=times,
            labels=self.label,
            time_to_task=self.time_to_task,
            std_time_to_task=self.std_time_to_task,
            maximum_time=self.maximum_time,
            seed=self.seed,
        )
        mask_records = [
            (x < y).type(th.float32) for x, y in zip(times, end_of_records)
        ]
        return [x for x in labels], mask_records

    def build_discretised_labels(
        self,
        events: list,
        times: list,
    ) -> (th.Tensor, th.Tensor):
        """
        Build discretised labels.

        Args:
            events (list): List of events.
            times (list): List of times.

        Returns:
            (th.Tensor, th.Tensor): Two tensors of labels and tasks
        """
        labels_, end_of_records = create_labels(
            events=events,
            event_times=times,
            labels=self.label,
            time_to_task=self.time_to_task,
            std_time_to_task=self.std_time_to_task,
            maximum_time=self.maximum_time,
            seed=self.seed,
        )
        idx = (
            (end_of_records / self.granularity)
            .long()
            .clamp(max=self.maximum_time / self.granularity)
        )
        mask_records = th.zeros(
            (len(events), int(self.maximum_time / self.granularity) + 1, 1)
        ).type(th.float32)
        for i in range(len(mask_records)):
            if idx[i] >= 0:
                mask_records[i][: idx[i]] = th.ones((idx[i], 1))

        return labels_, mask_records
