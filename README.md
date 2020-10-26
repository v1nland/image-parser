# Image Parser

Image Parser is a simple Python program that turns text to image and vice-versa

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all requirements (listed on requirements.txt).

```bash
pip3 install -r requirements.txt
```

## Usage

The following command executes "vec2image". Vec2Image reads the default model named "corona.w2v" and looks for words file "words.txt". Then it normalizes word's coordinates and generates an grayscaled image named "vec2image.png".

```bash
sudo python3 image_parser.py -c vec2image -o vec2image
```

The following command executes "image2vec". Image2Vec reads the default image named "vec2image.png", then converts it to an grayscale image and prints the coordinates of the image. Finally, it generates an image based on this new coordinates that is very similar to the first one named "image2vec.png".

```bash
sudo python3 image_parser.py -c image2vec -o image2vec
```

## Commands

Example of the help command:

```
usage: image_parser.py [-h] [--model MODEL] --command COMMAND [--input INPUT]
                       --output OUTPUT [--size SIZE] [--words WORDS]

optional arguments:
  -h, --help                     show this help message and exit
  --model MODEL, -m MODEL        Specify the model name (default = corona.w2v)
  --command COMMAND, -c COMMAND  Command to specify the mode (vec2image/image2vec)
  --input INPUT, -i INPUT        Input file name for image2vec (default = vec2image)
  --output OUTPUT, -o OUTPUT     Output file name
  --size SIZE, -s SIZE           Size of the image generated (default = 16)
  --words WORDS, -w WORDS        Words file name to read (default = words)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
