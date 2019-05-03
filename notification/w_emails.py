#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
import ssl
# from global_dict.w_global import gbl_get_value


def send_emails(model_id):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "notification.convez@gmail.com"  # Enter your address
    receiver_email = "wchen376@wisc.edu"  # Enter receiver address
    password = "nosugar591"
    message = "Subject: notification from WIMR_P6000 of " + model_id

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

    # server = smtplib.SMTP_SSL(smtp_server, port)
    # server.login(sender_email, password)
    # server.sendmail(
    #     sender_email,
    #     receiver_email,
    #     message)
    # server.quit()

