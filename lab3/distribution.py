from enum import Enum


class DistributionType(Enum):
    CONTINUOUS = 1
    DISCRETE = 2


class Distribution(Enum):
    NORMAL = 1
    CAUCHY = 2
    LAPLACE = 3
    POISSON = 4
    UNIFORM = 5

    @staticmethod
    def in_str(distribution):
        str_distribution = {
            Distribution.NORMAL: 'Normal',
            Distribution.CAUCHY: 'Cauchy',
            Distribution.LAPLACE: 'Laplace',
            Distribution.POISSON: 'Poisson',
            Distribution.UNIFORM: 'Uniform'
        }
        try:
            str = str_distribution[distribution]
            return str
        except KeyError as e:
            raise ValueError('No key with this value')
            return ''

    @staticmethod
    def distribution_type(distribution):
        type_distribution = {
            Distribution.NORMAL: DistributionType.CONTINUOUS,
            Distribution.CAUCHY: DistributionType.CONTINUOUS,
            Distribution.LAPLACE: DistributionType.CONTINUOUS,
            Distribution.POISSON: DistributionType.DISCRETE,
            Distribution.UNIFORM: DistributionType.CONTINUOUS
        }
        try:
            type = type_distribution[distribution]
            return type
        except KeyError as e:
            raise ValueError('No key with this value')
            return None