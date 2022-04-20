# Forum Virium Street Object Recognition

## Installation
```shell
pip install -r requirements.txt
```

## Jupyter notebooks
Execute
```shell
jupyter-lab
```
to open jupyter-lab environment.

### Training
Training the model can be done with `train.ipynb`.

There are two options at the moment (depending on which cells you execute):
    - use whole images (use "For image data" cells).
        - these are not perspective corrected
        - these have a lot of patches with no target classes in them
        - this takes a long time (~3-4 hours per epoch on Emblica bigrig).
    - use prepared .npzs
        - are perspectice corrected and sampled so there are more target classes in them
        - takes shorter time (~15mins per epoch on Emblica bigrig)
So unless you really want to try something, use .npzs :)
You can generate more of them by using `rearrange_dataset.ipynb`

### Prediction
There are multiple prediction pipeline options: one for doing prediction on a folder and three for predicting on a single image.

You can use `predict.ipynb` to do inference on a folder. The notebook has the prediction pipeline first and exploration/checkups later, and should be commented.

For inference on a single image file, you have, again, multiple options.
`predict-single-file.ipynb` allows for most flexibility and exploring.
`python predict.py <target_image> <output>` is for commandline.
There is also a docker image which can be used! Build it (you'll need a trained model, change the `COPY` statement accordingly in `Dockerfile` provided).
Then you can use a command similar to the example below to run inference on a single image.

Example docker run command:
```bash
docker run \
-v <absolute_path_to_folder_containing_target_images>:/app/input \
-v <absolute_path_to_output_folder>:/app/output \
forum-virium:latest \
python predict.py input/<target_image> output/<output_folder>
```

## Data
Data source of truth is https://cvat.apps.emblica.com/projects/2

## Training instructions (in finnish)
Tässä vielä s3uri annotoituihin kuviin: s3://forum-virium-street-vision-results/labels/. Lataaminen onnistuu helposti esimerkiksi käyttämällä s3cli:tä. Tämän jälkeen target_directory:stä pitää löytyä kokonaiset 360-kuvat ja label_directory:stä annotaatiokuvat, ja treenaaminen lähtee käyntiin.

Suosittelen kuitenkin, että käytette notebookin tarjoamaa .npz-mahdollisuutta. Sama data, mutta se täytyy esikäsitellä kahdessa vaiheessa. Ensimmäisessä vaiheessa korjataan 360-kuvan perspektiivivääristymä ja toisessa valitaan aineisto. 

Ensimmäistä vaihetta varten ajakaa perspective_to_equirectangle.ipynb-notebook. Kaksi ensimmäistä solua sisältävät olennaiset funktiomäärittelyt, sen jälkeen tulee muutama testisolu, mutta olennaisesti haluatte ajaa panorama2cube-funktion, joka ottaa ensimmäisenä argumenttinaan labelit sisältävän hakemiston, toisena sen hakemiston mihin perspektiivikorjatut labelit tulisi pistää, kolmantena koulutuskuvat sisältävän hakemiston, neljäntenä sen hakemiston mihin perspektiivikorjatut koulutuskuvat tulisi pistää ja viidentenä kuvauksen label-tiedostonimistä koulutustiedostonimiin. Viides argumentti on hieman outo, mutta sitä tarvitaan sillä labelointiin käytetty CVAT sekoitti tiedostojen nimiä.

Nämä ajettuanne teillä pitäisi olla output-direissä perspektiivikorjatut kuvat. Ellen väärin muista, jokaisesta koulutuskuvasta ja annotaatiosta luodaan kuusi “potrettia”. Seuraavaksi datasetistä pitää valita ne kuvat, joissa esiintyy mielenkiinnon kohteena olevia luokkia sekä saattaa ne kuvat tarpeeksi pieniksi osiksi ja pakata .npz-muotoon. Tämä tehdään ajamalla rearrange_dataset.ipynb:tä. Muistaakseni ainut tämän notebookin ajamisessa muistettava asia on, ettei se luo tarvitsemiaan hakemistoja, vaan se jätetään käyttäjän vastuulle. 

Tämän jälkeen outdir:n osoittamassa hakemistossa pitäisi olla suhteellisen monta tiedostoa, joiden nimet mukailevat muotoa "20210914-kruununhaka_GSAD0186_left-00.npz”.

Nyt on aika siirtyä train.ipynb:n puolelle. Ensimmäisen import-solun suorittamisen jälkeen voi hypätä suoraan “For .npz data”-solun ja sitä seuraavien solujen suorittamiseen.
