## Train
`python train.py -e 50 -g 0 -z 32 -s result -b 64`

## Test
`python test.py -e result/epoch50.enc -d result/epoch50.dec -s result/figs -g 0 -z 32`

## plot logs
`python view_log.py -j result/log.json -s result/figs`

## 復元結果
入力画像
![入力サンプル](https://github.com/wrb0312/for_labo/blob/master/vae/result/figs/result_input.jpg)

復元画像
![復元サンプル](https://github.com/wrb0312/for_labo/blob/master/vae/result/figs/result_reco.jpg)


## 生成サンプル
![生成サンプル](https://github.com/wrb0312/for_labo/blob/master/vae/result/figs/result_random.jpg)

## 低次元多様体上での変化
![画像AB](https://github.com/wrb0312/for_labo/blob/master/vae/result/figs/AB.jpg)

低次元空間において、AとBの間に存在する画像を可視化した.
![AからBへの変化](https://github.com/wrb0312/for_labo/blob/master/vae/result/figs/latent_A2B.jpg)

## 誤差の推移
![誤差変化](https://github.com/wrb0312/for_labo/blob/master/vae/result/figs/loss.jpg)