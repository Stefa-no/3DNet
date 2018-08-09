#!/usr/bin/env python
import os
from pprint import pprint

import time

import datetime
from telepot.loop import MessageLoop

import telepot

API_TOKEN = '452879752:AAFjodOl4xWCn89FxvE_f__kzkfxzFtc17g'
model = ''

script_to_run = '''python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --filename {} {}'''

def parseStuff(text):
	all_params = ""
	model = ''
	if ("-model" in text):
		index_of_first = text.find("-model") + 7
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--model {} ".format(text[index_of_first: index_of_last])
		model = text[index_of_first: index_of_last-1]
	all_params = text.replace('-', '--')
	return all_params, model
	#text = text.encode('utf-8')
'''	if ("-aug" in text):
		index_of_first = text.find("-aug") + 5
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--aug {} ".format(text[index_of_first: index_of_last])
	if ("-cuda" in text):
		all_params += "--cuda "
	if ("-pretrained" in text):
		all_params += "--pretrained "
	if ("-batch_size" in text):
		index_of_first = text.find("-batch_size") + 12
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--batch_size {} ".format(text[index_of_first: index_of_last])		'''

'''	if ("-lr" in text):
		index_of_first = text.find("-lr") + 4
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--lr {} ".format(text[index_of_first: index_of_last])
	if ("-epochs" in text):
		index_of_first = text.find("-epochs") + 8
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--epochs {} ".format(text[index_of_first: index_of_last])
	if ("-optimizer" in text):
		index_of_first = text.find("-tag") + 5
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--tag {} ".format(text[index_of_first: index_of_last])
	if ("-tag" in text):
		index_of_first = text.find("-tag") + 5
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--tag {} ".format(text[index_of_first: index_of_last])
	if ("-tag" in text):
		index_of_first = text.find("-tag") + 5
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--tag {} ".format(text[index_of_first: index_of_last])
	if ("-tag" in text):
		index_of_first = text.find("-tag") + 5
		index_of_last = text.find(" ", index_of_first) + 1
		if (index_of_last == 0): index_of_last = len(text)
		all_params += "--tag {} ".format(text[index_of_first: index_of_last])
'''	
	



def handle(msg):
	content_type, chat_type, chat_id = telepot.glance(msg)
	all_params = ""	
	today = datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p");
	try:
		all_params, model = parseStuff(msg['text'])
		
		default_tag = model +'_'+ today;
		file_name ='.log'
		
		#file_name = all_params.replace('--','').replace(' ', '_')
		#file_name += "_"+ today
		print(default_tag)
		bot.sendMessage(chat_id, "Running...")
		os.system(script_to_run.format(default_tag, all_params))
		bot.sendDocument(chat_id, open("./logs/" + default_tag + ".log", "r"))
		bot.sendMessage(chat_id, "Finish!")
	except Exception as e:
		bot.sendMessage(chat_id, "{}".format(e))
		bot.sendMessage(chat_id, "Dead!")


if __name__== "__main__":
	bot = telepot.Bot(API_TOKEN)
	
	MessageLoop(bot, handle).run_as_thread()
	print("listening...")
	while 1:
		time.sleep(10)
