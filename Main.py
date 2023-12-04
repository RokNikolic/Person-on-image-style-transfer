import os
import matplotlib.pylab as plt
import cvzone
import time
import random
import smtplib
import ssl
import json
from email.message import EmailMessage
import filetype
from MediaPipe import *
from StyleTransfer_TensorFlow import *

# Get email config from json file
try:
    with open("Email_config.json", "r") as read_file:
        email_data = json.load(read_file)
        email_flag = email_data["send_flag"]
        sender_email = email_data["sender_email"]
        sender_email_app_password = email_data["sender_app_password"]
        receiver_email = email_data["receiver_email"]
except Exception as e:
    print(f"Error getting email config data from file: {e}")


def import_person_image():
    file_name = os.listdir("picture of person")[0]
    if file_name is None:
        raise ValueError("No picture of person")
    return file_name


def import_background():
    file_name = os.listdir("background")[0]
    if file_name is None:
        raise ValueError("No background")
    return file_name


def overlay_person(image_person, background_file_name):
    background = cv2.imread(f"background/{background_file_name}")
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    image_person = image_resize(image_person, 500)

    # all possible positions
    potential_positions = [0, 650, 1000, 1000, 1100]
    pick = random.randint(0, 4)

    # overlay person at position
    position = [potential_positions[pick], background_rgb.shape[0]-image_person.shape[0]-5]
    overlaid_image = cvzone.overlayPNG(background_rgb, image_person, position)

    return overlaid_image


def image_resize(image, width=None, height=None):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


def send_email(image):
    # Get image from file
    image_data = image.read()
    image_type = filetype.guess(image.name)
    image_name = image.name

    # Create message
    message = EmailMessage()
    message['Subject'] = "Your image on picture"  # Defining email subject
    message['From'] = sender_email  # Defining sender email
    message['To'] = receiver_email  # Defining receiver email
    message.set_content(r"Were sending your image on a background")  # Defining email body
    # Add photo
    message.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
        server.login(sender_email, sender_email_app_password)
        server.send_message(message)


if __name__ == '__main__':
    file_person = import_person_image()
    file_background = import_background()

    background_mask = remove_background(file_person)
    transferred_image_person = transfer_style(file_person, file_background)
    transparent_background = np.zeros(transferred_image_person.shape, dtype=np.uint8)

    # combine person and background image using the mask
    final_image_person = np.where(background_mask, transferred_image_person, transparent_background)

    # make alpha channel with transparent background
    alpha_image_person = cv2.cvtColor(final_image_person, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(alpha_image_person, 0, 255, cv2.THRESH_BINARY)

    # add alpha channel to image
    r, g, b = cv2.split(final_image_person)
    a = np.ones(r.shape, dtype=np.uint8) * 255
    final_image_person = cv2.merge((r, g, b, alpha))

    # put person on image
    final_image = overlay_person(final_image_person, file_background)

    # show final image
    plt.figure()
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(final_image)
    plt.axis('off')

    # save image to folder, comment to disable saving
    name_of_image = f"final images/person_on_image_{time.strftime('%H.%M.%S', time.localtime())}.png"
    plt.savefig(name_of_image)

    # Send picture over email
    if email_flag == "True":
        try:
            with open(name_of_image, 'rb') as image_file:
                send_email(image_file)
        except Exception as e:
            print(f"Error when sending email: {e}")

    # Finally show image, this has to be the last line of code, because it wipes the image from program
    plt.show()
