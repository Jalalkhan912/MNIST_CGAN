# CGAN Handwritten Digit Generator

A web application that generates realistic handwritten digits (0-9) using a Conditional Generative Adversarial Network (CGAN). Users can specify which digit they want to generate and how many samples to create, all through an intuitive Streamlit interface.

## 🎯 Features

- **Conditional Generation**: Generate specific digits on demand (0-9)
- **Batch Generation**: Create multiple samples at once (1-25 images)
- **Real-time Visualization**: Instantly view generated digit images
- **Interactive Interface**: Simple web-based UI for easy interaction
- **High-Quality Output**: CGAN-generated realistic handwritten digits

## 🧠 Model Architecture

This application uses a **Conditional GAN (CGAN)** trained on the MNIST dataset:

### Generator Network
- **Input**: Random noise (100-dimensional) + digit label
- **Output**: 28x28 grayscale handwritten digit images
- **Architecture**: Conditional deep neural network that learns to generate digit-specific features

### Key Components
- **Latent Dimension**: 100-dimensional noise vector
- **Conditioning**: Digit labels (0-9) guide the generation process
- **Output Format**: Normalized grayscale images (28x28 pixels)

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Pre-trained CGAN model file (`generator.h5`)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jalalkhan912/MNIST_CGAN.git
   cd cgan-digit-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file is present**
   - Place your trained `generator.h5` model file in the project root directory
   - The model should be a trained CGAN generator compatible with MNIST digit generation

4. **Run the application**
   ```bash
   streamlit run CGAN_digit_app.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📋 Requirements

```
tensorflow
numpy
matplotlib
streamlit
```

## 📖 How to Use

### 🎨 Generating Digits

1. **Launch the application** using the command above
2. **Select target digit**: Choose any digit from 0-9 using the number input
3. **Set sample count**: Specify how many images to generate (1-25)
4. **Click "Generate"**: The app will create and display your requested digits
5. **View results**: Generated images appear with the specified digit label

### 🎛️ Interface Controls

- **Digit Selection**: Number input field (0-9)
- **Sample Count**: Slider or number input (1-25 samples)
- **Generate Button**: Triggers the generation process
- **Real-time Display**: Generated images appear instantly

## 🎨 Use Cases

### 📚 **Education & Research**
- Demonstrate GAN capabilities in educational settings
- Research data augmentation for digit recognition
- Study generative model behavior and outputs

### 🧪 **Data Science & ML**
- Augment MNIST datasets with synthetic samples
- Test digit recognition models with generated data
- Explore conditional generation techniques

### 🎯 **Creative Applications**
- Generate custom handwritten-style digits
- Create synthetic datasets for testing
- Prototype digit-based design elements

## 🔧 Technical Implementation

### Core Functions

#### `generate_specific_digits(generator, digit, num_samples)`
- **Purpose**: Generates specified digits using the trained model
- **Parameters**: 
  - `generator`: Loaded CGAN generator model
  - `digit`: Target digit (0-9)
  - `num_samples`: Number of images to generate
- **Returns**: Generated image tensors

#### `plot_generated_digits(images, digit)`
- **Purpose**: Creates matplotlib visualization of generated digits
- **Parameters**:
  - `images`: Generated image tensors
  - `digit`: Digit label for titling
- **Returns**: BytesIO buffer containing the plot

### Data Flow

1. **User Input**: Digit selection and sample count
2. **Noise Generation**: Random 100D vectors created
3. **Label Creation**: Digit labels formatted for conditioning
4. **Model Inference**: Generator produces images from noise + labels
5. **Post-processing**: Images rescaled from [-1,1] to [0,1]
6. **Visualization**: Matplotlib creates grid display
7. **Display**: Streamlit shows the generated images

## 📁 Project Structure

```
cgan-digit-generator/
│
├── CGAN_digit_app.py      # Main Streamlit application
├── requirements.txt       # Python dependencies
├── generator.h5          # Pre-trained CGAN generator model
├── README.md            # Project documentation
└── assets/              # Optional: screenshots, examples
    └── example_outputs/ # Sample generated digits
```

## ⚙️ Configuration

### Model Parameters
- **Latent Dimension**: 100 (modify `latent_dim` variable)
- **Image Size**: 28x28 pixels (MNIST standard)
- **Color Space**: Grayscale
- **Value Range**: [0, 1] after rescaling

### Generation Parameters
- **Sample Range**: 1-25 images per generation
- **Digit Range**: 0-9 (standard digits)
- **Noise Distribution**: Normal distribution (mean=0, std=1)

## 🎭 Model Training (Optional)

If you need to train your own CGAN model:

### Training Data
- **Dataset**: MNIST handwritten digits (60,000 training samples)
- **Format**: 28x28 grayscale images with digit labels
- **Preprocessing**: Normalization to [-1, 1] range

### Architecture Requirements
- **Generator Input**: Noise vector (100D) + one-hot encoded labels
- **Generator Output**: 28x28x1 images
- **Training**: Standard GAN adversarial training with conditional inputs

### Model Saving
```python
# Save only the generator for inference
generator.save('generator.h5')
```

## 🐛 Troubleshooting

### Common Issues

**"Model file not found" error:**
- Ensure `generator.h5` is in the project root directory
- Verify the model file is not corrupted
- Check file permissions

**Poor generation quality:**
- Verify the model was properly trained on MNIST data
- Ensure the latent dimension matches your trained model
- Check if the model expects different input formats

**Memory errors:**
- Reduce the number of samples generated at once
- Close other applications using GPU/memory
- Consider using CPU-only inference for smaller models

**Slow generation:**
- Reduce batch size for generation
- Consider model optimization techniques
- Use GPU acceleration if available

## 🔬 Advanced Features

### Custom Modifications

#### Change Latent Dimension
```python
latent_dim = 128  # Modify to match your trained model
```

#### Adjust Generation Range
```python
# Modify in Streamlit components
max_value=50  # Allow up to 50 samples
```

#### Custom Visualization
```python
# Modify plot_generated_digits() for different layouts
fig, axs = plt.subplots(2, num_images//2, figsize=(10, 8))
```

## 📊 Performance Metrics

### Generation Speed
- **Typical**: 1-5 images/second (CPU)
- **With GPU**: 10-50 images/second
- **Depends on**: Model size, hardware, batch size

### Quality Assessment
- **Visual Quality**: Compare with real MNIST digits
- **Diversity**: Check variation across multiple generations
- **Conditioning**: Verify correct digit generation

## 🤝 Contributing

We welcome contributions to improve the CGAN Digit Generator!

### Ways to Contribute
- **Model Improvements**: Better architectures or training techniques
- **UI Enhancements**: More interactive features or visualizations
- **Performance**: Optimization for faster generation
- **Documentation**: Examples, tutorials, or guides

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test thoroughly
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Database**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow Team**: For the deep learning framework
- **Streamlit Team**: For the web application framework
- **GAN Research Community**: For foundational generative modeling work

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Jalalkhan912/MNIST_CGAN/issues)
- **Email**: jalalkhanscience@gmail.com

## 🚀 Future Enhancements

- [ ] **Interactive Training**: Web-based model training interface
- [ ] **Style Transfer**: Generate digits in different handwriting styles
- [ ] **Batch Download**: Export generated images as ZIP files
- [ ] **Model Comparison**: Compare different generator architectures
- [ ] **Custom Datasets**: Support for other digit datasets
- [ ] **API Endpoint**: RESTful API for programmatic access
- [ ] **Mobile Support**: Responsive design for mobile devices


[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/)
[![GAN](https://img.shields.io/badge/Model-CGAN-green.svg)](https://arxiv.org/abs/1411.1784)
