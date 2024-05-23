## import packages

import os
import time                                     
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import imghdr

## define function
def report_send_mail(image_path):
    '''
    This function sends mail
    '''
    label = "Eye Close"
    with open(image_path, 'rb') as f:
        img_data = f.read()
    fromaddr = "ragulajay321@gmail.com"
    toaddr = "ragulajay321@gmail.com"
    msg = MIMEMultipart() 
    msg['From'] = fromaddr 
    msg['To'] = toaddr 
    msg['Subject'] = "Alert"
    body = label
    msg.attach(MIMEText(body, 'plain'))  # attach plain text
    image = MIMEImage(img_data, name=os.path.basename(image_path))
    msg.attach(image) # attach image
    s = smtplib.SMTP('smtp.gmail.com',587) 
    s.starttls() 
    s.login(fromaddr, "ragulajay321")
    text = msg.as_string() 
    s.sendmail(fromaddr, toaddr, text) 
    s.quit()

report_send_mail('image.jpg')
