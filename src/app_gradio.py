import gradio as gr
from model_classifier import ImageClassifier

classifier = ImageClassifier('resnet50')

def classify_image(image, num_predictions):
    """Classify image and return results"""
    predictions = classifier.predict(image, top_k=num_predictions)
    
    return {pred[0]: float(pred[1]) for pred in predictions}

demo = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(1, 10, value=5, step=1, label="Number of Predictions")
    ],
    outputs=gr.Label(num_top_classes=10, label="Predictions"),
    title="🖼️ Image Classification with Deep Learning",
    description="Upload an image to classify it using a pre-trained ResNet model",
    examples=[
        ["example1.jpg", 5],
        ["example2.jpg", 5]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)