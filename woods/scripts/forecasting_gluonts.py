from typing import NamedTuple, Optional, Iterable, Dict, Any, List, Tuple
import urllib
import os
import re

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gluonts.torch.util import weighted_average
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood

from datasets import load_dataset

import holidays
import datetime

from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
    InstanceSampler,
)
from gluonts.torch.util import (
    IterableDataset,
)
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import get_lags_for_frequency
from gluonts.dataset.common import Dataset, ListDataset
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.torch.modules.feature import FeatureEmbedder

from gluonts.core.component import validated

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]

class Electricity:

    def __init__(self):

        # Data property
        self.freq = "1H"
        self.num_feat_static_cat = 0
        self.num_feat_dynamic_real = 0
        self.num_feat_static_real = 0

        # Task information
        self.prediction_length = 24
        self.distr_output = StudentTOutput()        
        self.time_features = time_features_from_frequency_str(self.freq)
        print(self.time_features)

        # Context info
        self.context_length = 7*24
        self.lags_seq = get_lags_for_frequency(self.freq)
        self._past_length = self.context_length + max(self.lags_seq)

        # Training parameters
        self.batch_size = 50
        self.num_batches_per_epoch = 100

        # Sampler
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=self.prediction_length
        )
        self.validation_sampler = ValidationSplitSampler(
            min_future=self.prediction_length
        )

        # Scaler
        self.scaler = MeanScaler(dim=1, keepdim=True)

        # Embedding
        self.cardinality = [320]
        self.embedding_dimension = [5]
        self.embedder = FeatureEmbedder(
            cardinalities=self.cardinality,
            embedding_dims=self.embedding_dimension,
        )

        self.data = load_dataset('electricity_load_diagrams','lstnet')
        train_dataset = ListDataset(self.data['train'], freq=self.freq)
        validation_dataset = ListDataset(self.data['validation'], freq=self.freq)
        test_dataset = ListDataset(self.data['test'], freq=self.freq)

        self.transform = self.create_transformation()
        train_transformed = self.transform.apply(train_dataset, is_train=True)
        validation_transformed = self.transform.apply(validation_dataset, is_train=False)
        test_transformed = self.transform.apply(test_dataset, is_train=False)

        self.training_data_loader = self.create_training_data_loader(
            train_transformed,
            num_workers=0,
        )
        self.validation_data_loader = self.create_validation_data_loader(
            validation_transformed,
            num_workers=0,
        )
        self.test_data_loader = self.create_test_data_loader(
            test_transformed,
            num_workers=0,
        )

    def create_transformation(self) -> Transformation:
            remove_field_names = []
            if self.num_feat_static_real == 0:
                remove_field_names.append(FieldName.FEAT_STATIC_REAL)
            if self.num_feat_dynamic_real == 0:
                remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

            return Chain(
                [RemoveFields(field_names=remove_field_names)]
                + (
                    [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                    if not self.num_feat_static_cat > 0
                    else []
                )
                + (
                    [
                        SetField(
                            output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                        )
                    ]
                    if not self.num_feat_static_real > 0
                    else []
                )
                + [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT,
                        expected_ndim=1,
                        dtype=int,
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    ),
                    AsNumpyArray(
                        field=FieldName.TARGET,
                        # in the following line, we add 1 for the time dimension
                        expected_ndim=1 + len(self.distr_output.event_shape),
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                    ),
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=self.time_features,
                        pred_length=self.prediction_length,
                    ),
                    AddAgeFeature(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_AGE,
                        pred_length=self.prediction_length,
                        log_scale=True,
                    ),
                    VstackFeatures(
                        output_field=FieldName.FEAT_TIME,
                        input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                        + (
                            [FieldName.FEAT_DYNAMIC_REAL]
                            if self.num_feat_dynamic_real > 0
                            else []
                        ),
                    ),
                ]
            )

    def _create_instance_splitter(
            self, mode: str
        ):
            assert mode in ["training", "validation", "test"]

            instance_sampler = {
                "training": self.train_sampler,
                "validation": self.validation_sampler,
                "test": TestSplitSampler(),
            }[mode]

            return InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self._past_length,
                future_length=self.prediction_length,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
                dummy_value=self.distr_output.value_in_support,
            )


    def create_training_data_loader(
        self,
        data: Dataset,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        print(kwargs)
        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_test_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "test"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

class training_domain_sampler(InstanceSampler):

    axis: int = -1
    min_past: int = 0
    min_future: int = 0

    domain_idx: list = []
    
    num_instances: float
    total_length: int = 0
    n: int = 0

    def set_attribute(self, domain, set_holidays, start, max_length=0):
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        if domain == 'All':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                holidays_idx.append(idx)

            self.domain_idx = holidays_idx

        if domain == 'Holidays':
            holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time in set_holidays or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx)
                # if running_time.day in [24,25,26] and running_time.month in [12]:
                #     holidays_idx.append(idx)

            self.domain_idx = holidays_idx
        
        if domain == 'Non-holidays':
            non_holidays_idx = []
            for idx in range(max_length):
                running_time += min_increment
                if running_time not in set_holidays and running_time + day_increment not in set_holidays:
                    non_holidays_idx.append(idx)

            self.domain_idx = non_holidays_idx

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        in_range_idx = np.array(self.domain_idx)
        in_range_idx = in_range_idx[in_range_idx > a]
        in_range_idx = in_range_idx[in_range_idx < b]
        window_size = len(in_range_idx)

        if window_size <= 0:
            return np.array([], dtype=int)

        self.n += 1
        self.total_length += window_size
        avg_length = self.total_length / self.n

        if avg_length <= 0:
            return np.array([], dtype=int)

        p = self.num_instances / avg_length
        (indices,) = np.where(np.random.random_sample(window_size) < p)
        return in_range_idx[indices] + a

class evaluation_domain_sampler(InstanceSampler):

    axis: int = -1
    min_past: int = 0
    min_future: int = 0

    domain_idx: list = []
    
    num_instances: float
    total_length: int = 0
    n: int = 0

    start_idx: int

    def set_attribute(self, domain, set_holidays, start, max_length=0):
        
        min_increment = datetime.timedelta(minutes=30)
        day_increment = datetime.timedelta(days=1)
        running_time = start

        if domain == 'Holidays':
            holidays_idx = []
            for idx in range(int(max_length / 48)):
                running_time += day_increment
                if running_time in set_holidays or running_time + day_increment in set_holidays:
                    holidays_idx.append(idx * 48)
                # if running_time.day in [24,25,26] and running_time.month in [12]:
                #     holidays_idx.append(idx * 48)

            self.domain_idx = holidays_idx
        
        if domain == 'Non-holidays':
            non_holidays_idx = []
            for idx in range(int(max_length / 48)):
                running_time += day_increment
                if running_time not in set_holidays and running_time + day_increment not in set_holidays:
                    non_holidays_idx.append(idx * 48)

            self.domain_idx = non_holidays_idx

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        return (
            self.min_past,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        in_range_idx = np.array(self.domain_idx)
        in_range_idx = in_range_idx[in_range_idx > a]
        in_range_idx = in_range_idx[in_range_idx < b]
        in_range_idx = in_range_idx[in_range_idx > self.start_idx]
        window_size = len(in_range_idx)

        if window_size <= 0:
            return np.array([], dtype=int)

        return in_range_idx + a

class ChristmasHolidays(holidays.HolidayBase):

    def _populate(self, year):
        self[datetime.date(year, 12, 23)] = "Pre-Christmas Eve"
        self[datetime.date(year, 12, 24)] = "Christmas Eve"
        self[datetime.date(year, 12, 25)] = "Christmas"
        self[datetime.date(year, 12, 26)] = "Post-Christmas"
        self[datetime.date(year, 12, 26)] = "Christmas Break 1"
        self[datetime.date(year, 12, 27)] = "Christmas Break 2"
        self[datetime.date(year, 12, 28)] = "Christmas Break 3"
        self[datetime.date(year, 12, 29)] = "Christmas Break 4"
        self[datetime.date(year, 12, 30)] = "Christmas Break 5"
        self[datetime.date(year, 12, 31)] = "New Years eve"
        self[datetime.date(year, 1, 1)] = "New Years"
        self[datetime.date(year, 1, 2)] = "Post-New Years"

class AusElectricity:

    def __init__(self):

        # Domain property
        set_holidays = ChristmasHolidays()

        # Data property
        self.freq = "30T"
        self.num_feat_static_cat = 0
        self.num_feat_dynamic_real = 0
        self.num_feat_static_real = 0

        # Task information
        self.prediction_length = 48
        self.distr_output = StudentTOutput()
        self.time_features = time_features_from_frequency_str(self.freq)

        # Context info
        self.context_length = 7*self.prediction_length
        self.lags_seq = get_lags_for_frequency(self.freq)
        self._past_length = self.context_length + max(self.lags_seq)

        # Training parameters
        self.batch_size = 50
        self.num_batches_per_epoch = 100

        # Get dataset
        self.raw_data = load_dataset('monash_tsf','australian_electricity_demand')

        # Make splits
        val_first_idx = 192864
        test_first_idx = 210384
        target = self.raw_data['test']['target']
        start = self.raw_data['test']['start']

        # Create data split
        train_dataset = ListDataset(
            [
                {  
                    FieldName.TARGET: tgt[:val_first_idx],
                    FieldName.START: strt
                } for (tgt, strt) in zip(target, start)
            ],
            freq=self.freq
        )
        validation_dataset = ListDataset(
            [
                {
                    FieldName.TARGET: tgt[:test_first_idx],
                    FieldName.START: strt
                } for (tgt, strt) in zip(target, start)
            ], freq=self.freq
        )
        test_dataset = ListDataset(
            [
                {
                    FieldName.TARGET: tgt,
                    FieldName.START: strt
                } for (tgt, strt) in zip(target, start)
            ], freq=self.freq
        )

        # Samplers
        start_datetime = datetime.datetime(2002,1,1,0,0)
        max_ts_length = 240000
        self.training_all_sampler = training_domain_sampler(
            min_future=self.prediction_length, 
            num_instances=1.0
        )
        self.training_all_sampler.set_attribute(
            "All",
            set_holidays, 
            start_datetime, 
            max_length=max_ts_length
        )
        self.training_holiday_sampler = training_domain_sampler(
            min_future=self.prediction_length, 
            num_instances=1.0
        )
        self.training_holiday_sampler.set_attribute(
            "Holidays",
            set_holidays, 
            start_datetime, 
            max_length=max_ts_length
        )
        self.training_non_holiday_sampler = training_domain_sampler(
            min_future=self.prediction_length, 
            num_instances=1.0
        )
        self.training_non_holiday_sampler.set_attribute(
            "Non-holidays",
            set_holidays,
            start_datetime, 
            max_length=max_ts_length
        )

        self.evaluation_holiday_sampler = evaluation_domain_sampler(
            min_future=self.prediction_length, 
            num_instances=1.0,
            start_idx=val_first_idx
        )
        self.evaluation_holiday_sampler.set_attribute(
            "Holidays",
            set_holidays, 
            start_datetime, 
            max_length=max_ts_length
        )
        self.evaluation_non_holiday_sampler = evaluation_domain_sampler(
            min_future=self.prediction_length, 
            num_instances=1.0,
            start_idx=val_first_idx
        )
        self.evaluation_non_holiday_sampler.set_attribute(
            "Non-holidays",
            set_holidays,
            start_datetime, 
            max_length=max_ts_length
        )

        # Scaler
        self.scaler = MeanScaler(dim=1, keepdim=True)

        # Embedding
        self.cardinality = [320]
        self.embedding_dimension = [5]
        self.embedder = FeatureEmbedder(
            cardinalities=self.cardinality,
            embedding_dims=self.embedding_dimension,
        )

        # Create transformation
        self.transform = self.create_transformation()
        train_transformed = self.transform.apply(train_dataset, is_train=True)
        validation_transformed = self.transform.apply(validation_dataset, is_train=False)
        test_transformed = self.transform.apply(test_dataset, is_train=False)

        self.holiday_all_data_loader = self.create_training_data_loader(
            train_transformed,
            self.training_all_sampler,
            num_workers=0
        )
        self.holiday_training_data_loader = self.create_training_data_loader(
            train_transformed,
            self.training_holiday_sampler,
            num_workers=0
        )
        self.non_holiday_training_data_loader = self.create_training_data_loader(
            train_transformed,
            self.training_non_holiday_sampler,
            num_workers=0
        )
        self.holiday_validation_data_loader = self.create_evaluation_data_loader(
            validation_transformed,
            self.evaluation_holiday_sampler,
            num_workers=0
        )
        self.non_holiday_validation_data_loader = self.create_evaluation_data_loader(
            validation_transformed,
            self.evaluation_non_holiday_sampler,
            num_workers=0
        )
        self.holiday_test_data_loader = self.create_evaluation_data_loader(
            test_transformed,
            self.evaluation_holiday_sampler,
            num_workers=0
        )
        self.non_holiday_test_data_loader = self.create_evaluation_data_loader(
            test_transformed,
            self.evaluation_non_holiday_sampler,
            num_workers=0
        )

    def create_transformation(self) -> Transformation:
            remove_field_names = []
            if self.num_feat_static_real == 0:
                remove_field_names.append(FieldName.FEAT_STATIC_REAL)
            if self.num_feat_dynamic_real == 0:
                remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

            return Chain(
                [RemoveFields(field_names=remove_field_names)]
                + (
                    [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                    if not self.num_feat_static_cat > 0
                    else []
                )
                + (
                    [
                        SetField(
                            output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                        )
                    ]
                    if not self.num_feat_static_real > 0
                    else []
                )
                + [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT,
                        expected_ndim=1,
                        dtype=int,
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    ),
                    AsNumpyArray(
                        field=FieldName.TARGET,
                        # in the following line, we add 1 for the time dimension
                        expected_ndim=1 + len(self.distr_output.event_shape),
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                    ),
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=self.time_features,
                        pred_length=self.prediction_length,
                    ),
                    AddAgeFeature(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_AGE,
                        pred_length=self.prediction_length,
                        log_scale=True,
                    ),
                    VstackFeatures(
                        output_field=FieldName.FEAT_TIME,
                        input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                        + (
                            [FieldName.FEAT_DYNAMIC_REAL]
                            if self.num_feat_dynamic_real > 0
                            else []
                        ),
                    ),
                ]
            )

    def _create_instance_splitter(
            self, instance_sampler: InstanceSampler
        ):
            return InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self._past_length,
                future_length=self.prediction_length,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
                dummy_value=self.distr_output.value_in_support,
            )

    def create_training_data_loader(
        self,
        data: Dataset,
        instance_sampler: InstanceSampler,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(instance_sampler) + SelectFields(TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )
        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_evaluation_data_loader(
        self,
        data: Dataset,
        instance_sampler: InstanceSampler,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(instance_sampler) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

class LondonSmartMeters:

    def __init__(self):

        # Data property
        self.freq = "30T"
        self.num_feat_static_cat = 0
        self.num_feat_dynamic_real = 0
        self.num_feat_static_real = 0

        # Task information
        self.prediction_length = 60
        self.distr_output = StudentTOutput()        
        self.time_features = time_features_from_frequency_str(self.freq)

        # Context info
        self.context_length = 7*60
        self.lags_seq = get_lags_for_frequency(self.freq)
        self._past_length = self.context_length + max(self.lags_seq)

        # Training parameters
        self.batch_size = 50
        self.num_batches_per_epoch = 100

        # Sampler
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=self.prediction_length
        )
        self.validation_sampler = ValidationSplitSampler(
            min_future=self.prediction_length
        )

        # Scaler
        self.scaler = MeanScaler(dim=1, keepdim=True)

        # Embedding
        self.cardinality = [320]
        self.embedding_dimension = [5]
        self.embedder = FeatureEmbedder(
            cardinalities=self.cardinality,
            embedding_dims=self.embedding_dimension,
        )

        self.data = load_dataset('monash_tsf','london_smart_meters')
        train_dataset = ListDataset(self.data['train'], freq=self.freq)
        # validation_dataset = ListDataset(self.data['validation'], freq=self.freq)
        # print("salut")
        # test_dataset = ListDataset(self.data['test'], freq=self.freq)
        # print('aurevoir')

        self.transform = self.create_transformation()
        train_transformed = self.transform.apply(train_dataset, is_train=True)
        # validation_transformed = self.transform.apply(validation_dataset, is_train=False)
        # test_transformed = self.transform.apply(test_dataset, is_train=False)


        self.training_data_loader = self.create_training_data_loader(
            train_transformed,
            num_workers=0,
        )
        # self.validation_data_loader = self.create_validation_data_loader(
        #     validation_transformed,
        #     num_workers=0,
        # )
        # self.test_data_loader = self.create_test_data_loader(
        #     test_transformed,
        #     num_workers=0,
        # )


    def create_transformation(self) -> Transformation:
            remove_field_names = []
            if self.num_feat_static_real == 0:
                remove_field_names.append(FieldName.FEAT_STATIC_REAL)
            if self.num_feat_dynamic_real == 0:
                remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

            return Chain(
                [RemoveFields(field_names=remove_field_names)]
                + (
                    [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                    if not self.num_feat_static_cat > 0
                    else []
                )
                + (
                    [
                        SetField(
                            output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                        )
                    ]
                    if not self.num_feat_static_real > 0
                    else []
                )
                + [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT,
                        expected_ndim=1,
                        dtype=int,
                    ),
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    ),
                    AsNumpyArray(
                        field=FieldName.TARGET,
                        # in the following line, we add 1 for the time dimension
                        expected_ndim=1 + len(self.distr_output.event_shape),
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                    ),
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=self.time_features,
                        pred_length=self.prediction_length,
                    ),
                    AddAgeFeature(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_AGE,
                        pred_length=self.prediction_length,
                        log_scale=True,
                    ),
                    VstackFeatures(
                        output_field=FieldName.FEAT_TIME,
                        input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                        + (
                            [FieldName.FEAT_DYNAMIC_REAL]
                            if self.num_feat_dynamic_real > 0
                            else []
                        ),
                    ),
                ]
            )

    def _create_instance_splitter(
            self, mode: str
        ):
            assert mode in ["training", "validation", "test"]

            instance_sampler = {
                "training": self.train_sampler,
                "validation": self.validation_sampler,
                "test": TestSplitSampler(),
            }[mode]

            return InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self._past_length,
                future_length=self.prediction_length,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
                dummy_value=self.distr_output.value_in_support,
            )

    def create_training_data_loader(
        self,
        data: Dataset,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_test_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            "test"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

class TransformerModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        
        # transformer arguments
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        activation: str = "gelu",
        dropout: float = 0.1,

        # univariate input
        input_size: int = 1,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()
        
        self.input_size = input_size
       
        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
        
        # total feature size
        d_model = self.input_size * len(self.lags_seq) + self._number_of_features
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)

        print(self.input_size, len(self.lags_seq), self._number_of_features)
        print(d_model, nhead)
            
        # transformer enc-decoder and mask initializer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        
        # causal decoder tgt mask
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(prediction_length),
        )
        
    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 1  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)
    
    def get_lagged_subsequences(
        self,
        sequence: torch.Tensor,
        subsequences_length: int,
        shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        shift: int
            shift the lags by this amount back.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = [l - shift for l in self.lags_seq]

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def _check_shapes(
        self,
        prior_input: torch.Tensor,
        inputs: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> None:
        assert len(prior_input.shape) == len(inputs.shape)
        assert (
            len(prior_input.shape) == 2 and self.input_size == 1
        ) or prior_input.shape[2] == self.input_size
        assert (len(inputs.shape) == 2 and self.input_size == 1) or inputs.shape[
            -1
        ] == self.input_size
        assert (
            features is None or features.shape[2] == self._number_of_features
        ), f"{features.shape[2]}, expected {self._number_of_features}"
    
    
    def create_network_inputs(
        self, 
        feat_static_cat: torch.Tensor, 
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):        
        # time feature
        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # target
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, scale = self.scaler(context, observed_context)

        inputs = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )

        inputs_length = (
            self._past_length + self.prediction_length
            if future_target is not None
            else self._past_length
        )
        assert inputs.shape[1] == inputs_length
        
        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )
        
        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, time_feat.shape[1], -1
        )
        
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)
        
        # Lagged
        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs,
            subsequences_length=subsequences_length,
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], -1
        )

        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)
        
        return transformer_inputs, scale, static_feat
    
    def output_params(self, transformer_inputs):
        enc_input = transformer_inputs[:, :self.context_length, ...]
        dec_input = transformer_inputs[:, self.context_length:, ...]
        
        enc_out = self.transformer.encoder(
            enc_input
        )

        dec_output = self.transformer.decoder(
            dec_input,
            enc_out,
            tgt_mask=self.tgt_mask
        )

        
        return self.param_proj(dec_output)

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)
    
    def alt_forward(
        self,
        batch
    ):

        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]

        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]

        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        
        
        transformer_inputs, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        params = self.output_params(transformer_inputs)
        distr = self.output_distribution(params, scale)

        return distr, future_target

    # for prediction
    def forward(
        self,
        batch,
        # feat_static_cat: torch.Tensor,
        # feat_static_real: torch.Tensor,
        # past_time_feat: torch.Tensor,
        # past_target: torch.Tensor,
        # past_observed_values: torch.Tensor,
        # future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:

        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]

        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]

        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        
        
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples
            
        encoder_inputs, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
        )

        enc_out = self.transformer.encoder(encoder_inputs)
        
        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(
                repeats=self.num_parallel_samples, dim=0
            )
            / repeated_scale
        )
        
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_feat.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )
       
        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        future_samples = []
        
        # greedy decoding
        for k in range(self.prediction_length):            
            #self._check_shapes(repeated_past_target, next_sample, next_features)
            #sequence = torch.cat((repeated_past_target, next_sample), dim=1)
            
            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1+k,
                shift=1, 
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )
            
            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k+1]), dim=-1)

            output = self.transformer.decoder(decoder_input, repeated_enc_out)
            
            params = self.param_proj(output[:,-1:])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length)
            + self.target_shape,
        )

def get_minute(minute):
    return minute + 0.5
def get_hour(hour):
    return hour + 0.5
def get_day_of_year(time_feat):
    return (time_feat + 0.5) * 365 + 1
def get_year(year):
    return (np.power(10, year)-2.0)/17532

def plot_forecast(k, batch, pred):
    plt.figure()
    minutes = get_minute(torch.cat((batch['past_time_feat'][k,:,0], batch['future_time_feat'][k,:,0]), dim=0))
    hours = get_hour(torch.cat((batch['past_time_feat'][k,:,1], batch['future_time_feat'][k,:,1]), dim=0))
    days = get_day_of_year(torch.cat((batch['past_time_feat'][k,:,-2], batch['future_time_feat'][k,:,-2]), dim=0))
    years = get_year(torch.cat((batch['past_time_feat'][k,:,-1], batch['future_time_feat'][k,:,-1]), dim=0))
    date_time = [(datetime.datetime(2002 + int(year.item()),1,1) + datetime.timedelta(days=day.item(), hours=hour.item(), minutes=minu.item())).strftime('%Y-%m-%d') for year, day, hour, minu in zip(years, days, hours, minutes)]
    labels = [''] * len(date_time)
    labels[::40] = date_time[::40]
    ground_truth = torch.cat((batch['past_target'][k], batch['future_target'][k]), dim=0)
    full_pred = torch.cat((batch['past_target'][k], torch.mean(out[k,:,:], dim=0)), dim=0)
    plt.plot(ground_truth, 'b', label='Ground Truth')
    plt.plot(full_pred, 'r', label='Prediction')
    plt.axvline(x=batch['past_target'][k].shape[-1])
    plt.xticks(np.arange(len(labels)), labels)
    plt.xticks(rotation=60)
    # plt.plot(date_time, pred)
    plt.gcf().tight_layout()
    plt.show()

def main(lr):

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    electricity = AusElectricity()
    # electricity = LondonSmartMeters()

    iterable = zip(electricity.holiday_validation_data_loader, electricity.non_holiday_validation_data_loader)
    next(iterable)
    print("val")
    iterable = zip(electricity.holiday_training_data_loader, electricity.non_holiday_training_data_loader)
    next(iterable)
    print("train")

    model = TransformerModel(
            freq=electricity.freq,
            context_length=electricity.context_length,
            prediction_length=electricity.prediction_length,
            num_feat_dynamic_real=1 + electricity.num_feat_dynamic_real + len(electricity.time_features),
            num_feat_static_real=max(1, electricity.num_feat_static_real),
            num_feat_static_cat=max(1, electricity.num_feat_static_cat),
            cardinality=electricity.cardinality,
            embedding_dimension=electricity.embedding_dimension,

            # transformer arguments
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            activation='gelu',
            dropout=0.1,
            dim_feedforward=32,

            # univariate input
            input_size=1,
            distr_output=StudentTOutput(),
            lags_seq=get_lags_for_frequency(electricity.freq),
            scaling=True,
            num_parallel_samples=10,
        )

    model.to(device)

    holiday_loader = electricity.holiday_training_data_loader
    non_holiday_loader = electricity.non_holiday_training_data_loader

    optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=0,
        )

    loss_fn = NegativeLogLikelihood()
    eval_loss = nn.MSELoss()
    for e in range(20):
        print("EPOCH: ", e)
        for i, batch_loaders in enumerate(zip(holiday_loader, non_holiday_loader)):
            minibatches_device = {k: torch.cat((batch_loaders[0][k],batch_loaders[1][k]), dim=0).to(device) for k in batch_loaders[0]}

        # # for i, minibatches_device in enumerate(electricity.holiday_all_data_loader):
        #     # for k in range(minibatches_device['past_time_feat'].shape[0]):
        #     #     plt.figure()
        #     #     minutes = get_minute(torch.cat((minibatches_device['past_time_feat'][k,:,0], minibatches_device['future_time_feat'][k,:,0]), dim=0))
        #     #     hours = get_hour(torch.cat((minibatches_device['past_time_feat'][k,:,1], minibatches_device['future_time_feat'][k,:,1]), dim=0))
        #     #     days = get_day_of_year(torch.cat((minibatches_device['past_time_feat'][k,:,-2], minibatches_device['future_time_feat'][k,:,-2]), dim=0))
        #     #     years = get_year(torch.cat((minibatches_device['past_time_feat'][k,:,-1], minibatches_device['future_time_feat'][k,:,-1]), dim=0))
        #     #     date_time = [(datetime.datetime(2002 + int(year.item()),1,1) + datetime.timedelta(days=day.item(), hours=hour.item(), minutes=minu.item())).strftime('%Y-%m-%d') for year, day, hour, minu in zip(years, days, hours, minutes)]
        #     #     labels = [''] * len(date_time)
        #     #     labels[::40] = date_time[::40]
        #     #     ground_truth = torch.cat((minibatches_device['past_target'][k], minibatches_device['future_target'][k]), dim=0)
        #     #     # pred = torch.cat((torch.zeros_like(minibatches_device['past_target'][i]), torch.mean(out[i,:,:], dim=0)), dim=0)
        #     #     plt.plot(ground_truth)
        #     #     plt.xticks(np.arange(len(labels)), labels)
        #     #     plt.xticks(rotation=60)
        #     #     # plt.plot(date_time, pred)
        #     #     plt.gcf().tight_layout()
        #     #     plt.show()

            out, future_target = model.alt_forward(minibatches_device)

            loss = loss_fn(out, future_target)
            loss = weighted_average(loss, weights=loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print(i, loss)

        holiday_val_loss = []
        for i, val_batch in enumerate(electricity.holiday_validation_data_loader):

            val_batch = {k: val_batch[k].to(device) for k in val_batch}

            out = model(val_batch)
            mean_pred = torch.mean(out, dim=1)

            # print(mean_pred[0,:], val_batch['future_target'][0,:])
            # plot_forecast(0, val_batch, out)

            loss = eval_loss(mean_pred, val_batch['future_target'])
            loss = weighted_average(loss, weights=loss)

            holiday_val_loss.append(np.sqrt(loss.item()))

            # print("Holiday Val", i)

        non_holiday_val_loss = []
        for i, val_batch in enumerate(electricity.non_holiday_validation_data_loader):

            val_batch = {k: val_batch[k].to(device) for k in val_batch}

            out = model(val_batch)

            loss = eval_loss(torch.mean(out, dim=1), val_batch['future_target'])
            loss = weighted_average(loss, weights=loss)

            non_holiday_val_loss.append(np.sqrt(loss.item()))

            # print("Non-Holiday Val", i)

        print("Holiday Val loss", sum(holiday_val_loss) / len(holiday_val_loss))
        print("Non-Holiday Val loss", sum(non_holiday_val_loss) / len(non_holiday_val_loss))

    # test_loss = []
    # for i, val_batch in enumerate(electricity.test_data_loader):

    #     out, future_target = model.alt_forward(batch)

    #     loss = loss_fn(out, future_target)
    #     loss = weighted_average(loss, weights=loss)

    #     test_loss.append(loss)

    #     print("Test", i)

    # print("Test loss", sum(test_loss) / len(test_loss))


    # for batch in loader:
    #     out = model(batch)

    #     for i in range(batch['past_time_feat'].shape[0]):
    #         plt.figure()
    #         time = get_time(
    #             torch.cat((batch['past_time_feat'][i,:,0], batch['future_time_feat'][i,:,0]), dim=0),
    #             torch.cat((batch['past_time_feat'][i,:,1], batch['future_time_feat'][i,:,1]), dim=0)
    #         )
    #         days = get_day_of_year(torch.cat((batch['past_time_feat'][i,:,-2], batch['future_time_feat'][i,:,-2]), dim=0))
    #         date_time = time + days
    #         ground_truth = torch.cat((batch['past_target'][i], batch['future_target'][i]), dim=0)
    #         pred = torch.cat((torch.zeros_like(batch['past_target'][i]), torch.mean(out[i,:,:], dim=0)), dim=0)
    #         plt.plot(date_time, ground_truth)
    #         plt.plot(date_time, pred)
    #         plt.show()



if __name__ == '__main__':

    for lr in [1e-4, 5e-5]:

        print("RUNNING LR =", lr)
        main(lr)




























######### THRASH 
# class FRED(Multi_Domain_Dataset):
#     """ 
#     """
#     ## Training parameters
#     N_STEPS = 5001
#     CHECKPOINT_FREQ = 100

#     ## Dataset parameters
#     SETUP = 'source'
#     TASK = 'regression'
#     SEQ_LEN = 500
#     PRED_TIME = [168]
#     INPUT_SHAPE = [6]
#     OUTPUT_SIZE = 1
#     frequency = '1M'
#     perdition_length = 24

#     ## Environment parameters
#     ENVS = [None]
#     SWEEP_ENVS = list(range(len(ENVS)))

#     def __init__(self, flags, training_hparams):
#         """ Dataset constructor function

#         Args:
#             flags (argparse.Namespace): argparse of training arguments
#             training_hparams (dict): dictionnary of training hyper parameters coming from the hyperparams.py file
#         """

#         lag = 15
#         external_forecast_horizon = 12

#         BASE_DIR = "TSForecasting"

#         # The name of the column containing time series values after loading data from the .tsf file into a dataframe
#         VALUE_COL_NAME = "series_value"

#         # The name of the column containing timestamps after loading data from the .tsf file into a dataframe
#         TIME_COL_NAME = "start_timestamp"

#         # Seasonality values corresponding with the frequencies: minutely, 10_minutes, half_hourly, hourly, daily, weekly, monthly, quarterly and yearly
#         # Consider multiple seasonalities for frequencies less than daily
#         SEASONALITY_MAP = {
#         "minutely": [1440, 10080, 525960],
#         "10_minutes": [144, 1008, 52596],
#         "half_hourly": [48, 336, 17532],
#         "hourly": [24, 168, 8766],
#         "daily": 7,
#         "weekly": 365.25/7,
#         "monthly": 12,
#         "quarterly": 4,
#         "yearly": 1
#         }

#         # Frequencies used by GluonTS framework
#         FREQUENCY_MAP = {
#         "minutely": "1min",
#         "10_minutes": "10min",
#         "half_hourly": "30min",
#         "hourly": "1H",
#         "daily": "1D",
#         "weekly": "1W",
#         "monthly": "1M",
#         "quarterly": "1Q",
#         "yearly": "1Y"
#         }

#         df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe('/hdd/data/monash/fred_md_dataset.tsf')


#         train_series_list = []
#         test_series_list = []
#         train_series_full_list = []
#         test_series_full_list = []
#         final_forecasts = []

#         if frequency is not None:
#             freq = FREQUENCY_MAP[frequency]
#             seasonality = SEASONALITY_MAP[frequency]
#         else:
#             freq = "1Y"
#             seasonality = 1

#         if isinstance(seasonality, list):
#             seasonality = min(seasonality) # Use to calculate MASE

#         # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
#         if forecast_horizon is None:
#             if external_forecast_horizon is None:
#                 raise Exception("Please provide the required forecast horizon")
#             else:
#                 forecast_horizon = external_forecast_horizon

#         start_exec_time = datetime.now()

#         for index, row in df.iterrows():
#             if TIME_COL_NAME in df.columns:
#                 train_start_time = row[TIME_COL_NAME]
#             else:
#                 train_start_time = datetime.strptime('1900-01-01 00-00-00', '%Y-%m-%d %H-%M-%S') # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

#             series_data = row[VALUE_COL_NAME]

#             # Creating training and test series. Test series will be only used during evaluation
#             train_series_data = series_data[:len(series_data) - forecast_horizon]
#             test_series_data = series_data[(len(series_data) - forecast_horizon) : len(series_data)]

#             train_series_list.append(train_series_data)
#             test_series_list.append(test_series_data)

#             # We use full length training series to train the model as we do not tune hyperparameters
#             train_series_full_list.append({
#                 FieldName.TARGET: train_series_data,
#                 FieldName.START: pd.Timestamp(train_start_time, freq=freq)
#             })

#             test_series_full_list.append({
#                 FieldName.TARGET: series_data,
#                 FieldName.START: pd.Timestamp(train_start_time, freq=freq)
#             })

#         train_ds = ListDataset(train_series_full_list, freq=freq)
#         test_ds = ListDataset(test_series_full_list, freq=freq)

#         # trans = create_transformation()

#         # trans_ds = trans(train_ds)

#         estimator = TransformerEstimator(freq=freq,
#                                      context_length=lag,
#                                      prediction_length=forecast_horizon)

#         predictor = estimator.train(training_data=train_ds)

#         print(next(iter(train_ds)))

