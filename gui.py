from http.client import NETWORK_AUTHENTICATION_REQUIRED
from tkinter import *
from tkinter import filedialog
from tkinter.font import BOLD
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.models import load_model



# Root Initialization
root = Tk()
root.title('Wheat Leaf Disease Detection')
canvas_height = 100
canvas_width = 100

# Application Size
root.geometry("750x750")
root.minsize(750, 750)

def open_popup():
   about_popup = Toplevel(root)
   about_popup.geometry("250x250")
   about_popup.title("About")
   Label(about_popup, text= "Prepared by:\n\nAoanan, Marco Noel\nArambulo, Hans Xavier\nBuenaventura, Angelo\nCapulong, Mark\nEsguerra, William Gerrard\nEspañola, Joshua Ian\nFuentes, Rose Ann\nGlico, Jerico\nIlagan, Alira\nVerendia, John Justin\nYgrubay, Claude Jannin\n\nBSCS 3-2").grid()
   
   about_popup.rowconfigure(0, weight=1)
   about_popup.columnconfigure(0, weight=1)




# Main Frames
frame1 = Frame(root, highlightbackground="gray", highlightthickness=2)
frame2 = Frame(root, highlightbackground="gray", highlightthickness=2)
frame3 = Frame(frame2, highlightbackground="gray", highlightthickness=2)


# Main Grid Placement
frame1.grid(row = 0, column = 0, sticky = NSEW, pady = 5, padx = 5)
frame2.grid(row = 0, column = 1, sticky = NSEW, pady = 5, padx = 5)


# Main Grid Configuration
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)   # Frame 1
root.columnconfigure(1, weight=5)   # Frame 2

# Frame 1 Components
frame1_label = Frame(frame1)
frame1_image = Frame(frame1, highlightbackground="gray", highlightthickness=2)
frame1_image.configure(height=frame1_image.winfo_height(), width=frame1_image.winfo_height())
frame1_image.grid_propagate(0)
frame1_buttons = Frame(frame1, highlightbackground="gray", highlightthickness=2)
frame2_buttons = Frame(frame1, highlightbackground="gray", highlightthickness=2)



# Frame 2 Components
frame2_label_result = Frame(frame2)
frame2_label_result_desc = Frame(frame2)
frame2_label_accuracy = Frame(frame2)
frame2_label_accuracy_desc = Frame(frame2)

# Frame 1 Components Grid Placement
frame1_label.grid(row = 0, column = 0, sticky = NSEW, pady = 5, padx = 5)
frame1_image.grid(row = 1, column = 0, sticky = NSEW, pady = 5, padx = 5)
frame1_buttons.grid(row = 2, column = 0, sticky = NSEW, pady = 5, padx = 5)


# Frame 2 Components Grid Placement
frame2_label_result.grid(row = 0, column = 0, sticky = NSEW, pady = 5, padx = 5)
frame2_label_result_desc.grid(row = 0, column = 1, sticky = NSEW, pady = 5, padx = 5)
frame2_label_accuracy.grid(row = 0, column = 2, sticky = NSEW, pady = 5, padx = 5)
frame2_label_accuracy_desc.grid(row = 0, column = 3, sticky = NSEW, pady = 5, padx = 5)

# Frame 1 Components Grid Configuration
frame1.columnconfigure(0, weight=1)
frame1.rowconfigure(0, weight=1)   # Logo
frame1.rowconfigure(1, weight=25)   # Image
frame1.rowconfigure(2, weight=20)   # Buttons

# Label
label1 = Label( frame1_label, 
                text="Wheat Leaf Disease Detection",
                font=("Arial", 15, BOLD),)
label1.pack()
label2 = Label( frame2_label_result, 
                text="Output: ",
                font=("Arial", 15, BOLD),)
label2.pack()
label3 = Label( frame2_label_accuracy, 
                text="Accuracy: ",
                font=("Arial", 15, BOLD),)
label3.pack()
label4 = Label( frame2_label_result_desc, 
                text="___",
                font=("Arial", 15, BOLD),)
label4.pack()
label5 = Label( frame2_label_accuracy_desc, 
                text="___",
                font=("Arial", 15, BOLD),)
label5.pack()

# Function for Choosing Files
def open_file():
    global my_image
    global image_dir
    root.filename = filedialog.askopenfilename(initialdir="Desktop", 
                                               title="Choose Image File", 
                                               filetypes=(("JPG Files", "*.jpg"),
                                                          ("PNG Files", "*.png"),
                                                          ("JFIF Files", "*.jfif"),
                                                          ("All Files", "*.*")))
    my_label = Label(frame1_buttons, text=root.filename)
    my_label.grid(row = 0, columnspan=2, sticky = NSEW, pady = 5, padx = 5)
    my_image = Image.open(root.filename)
    image_dir = root.filename
    # Resize the image using resize() method
    
    resize_image = my_image.resize((int(my_image.height/my_image.width * frame1_image.winfo_height()), int(my_image.height/my_image.width * frame1_image.winfo_height())))
    
    
    my_image = ImageTk.PhotoImage(resize_image)
    
    # Display Image
    display_image = Label(frame1_image, image = my_image)
    display_image.grid(row=0, column=0, sticky=NSEW, pady=5, padx=5)
    display_image.rowconfigure(0, weight=1)
    display_image.columnconfigure(0, weight=1)
    
    #display_image.pack()

def check_image():
    img = image.load_img(image_dir, target_size=(224,224))
    img = np.asarray(img)
    #import model
    saved_model = load_model("vgg16_1.h5")

    #declare classes
    classes = ['Healthy', 'Septoria', 'Stripe Rust']
    #use model to predict the input
    #results is passed to output
    output = saved_model.predict(img)

    #get index of highest probability
    MaxPosition=np.argmax(output)

    #Get the highest probability
    probabilities=max(output[0])

    #assigns the class
    prediction_label=classes[MaxPosition]

    #displays the class name and percentage
    label4 = Label( frame2_label_result_desc, 
                text=prediction_label,
                font=("Arial", 15, BOLD),)
    label4.pack()
    label5 = Label( frame2_label_accuracy_desc, 
                    text=probabilities*100,
                    font=("Arial", 15, BOLD),)
    label5.pack()




# Frame1_Buttons Components
choose_button = Button(frame1_buttons, text="Choose Image", command=open_file)
about_button = Button(frame1_buttons, text="About", command=open_popup)
history_button = Button(frame1_buttons, text="History")
check_image_button = Button(frame1_buttons, text="Check Image", command=check_image)

# Buttons Grid Placement
choose_button.grid(row = 1, columnspan=2, sticky = NSEW, pady = 5, padx= 5)
about_button.grid(row = 2, column = 0, sticky = NSEW, pady = 5, padx= 5)
history_button.grid(row = 2, column = 1, sticky = NSEW, pady = 5, padx= 5)
check_image_button.grid(row = 3, column = 0, sticky = NSEW, pady = 5, padx= 5)

# Buttons Grid Configuration
frame1_buttons.columnconfigure(0, weight=1)
frame1_buttons.columnconfigure(1, weight=1)



# Image Placements
frame1_image.rowconfigure(0, weight=1)
frame1_image.columnconfigure(0, weight=1)



root.mainloop()