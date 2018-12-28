## Train
`python train.py -e 50 -g 0 -z 128 -s result -b 64`

## Test
`python test.py -m result/epoch50.gen -s result/figs -g 0`

## plot logs
`python view_log.py -j result/log.json -s result/figs`

## 生成サンプル
![生成サンプル](https://github.com/wrb0312/for_labo/blob/master/gan/result/figs/result.jpg)

## 低次元多様体上での変化
![画像AB](https://github.com/wrb0312/for_labo/blob/master/gan/result/figs/AB.jpg)

低次元空間において、AとBの間に存在する画像を可視化した.
![AからBへの変化](https://github.com/wrb0312/for_labo/blob/master/gan/result/figs/latent_A2B.jpg)