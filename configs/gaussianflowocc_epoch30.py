_base_ = ['./gaussianflowocc.py']

# NOTE: increase epoch from 18 to 30!
runner = dict(type='EpochBasedRunner', max_epochs=30)