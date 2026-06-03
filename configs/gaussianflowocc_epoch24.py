_base_ = ['./gaussianflowocc.py']

# NOTE: increase epoch from 18 to 24!
runner = dict(type='EpochBasedRunner', max_epochs=24)