## Train
`python train.py -e 50 -g 0 -z 128 -s result -b 64`

## Test
`python test.py -m result/epoch50.gen -s result/figs -g 0`

## plot logs
`python view_log.py -j result/log.json -s result/figs`
