# Image Parser

Image Parser is a simple Python program that turns text to image and vice-versa

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all requirements (listed on requirements.txt).

```bash
pip3 install -r requirements.txt
```

## Usage

The following command executes "vec2image". Vec2Image reads the model named "corona.w2v" and looks for word "generation". Then it normalizes word's coordinates and generates an grayscaled image named "vec2image.png".

```bash
sudo python3 image_parser.py vec2image
```

The following command executes "image2vec". Image2Vec reads the image named "vec2image.png", then converts it to an grayscale image and prints the coordinates of the image. Finally, it generates an image based on this new coordinates that is very similar to the first one.

```bash
sudo python3 image_parser.py image2vec
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
