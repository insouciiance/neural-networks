import cv2
import numpy as np

def recognize_video(model):
  video = cv2.VideoCapture('data/Cola3.mp4')
  frame_no = 0
  last_predictions = []
  last_result = False
  begin_time = []
  end_time = []
  while video.isOpened():
      ret, frame = video.read()
      if(ret):
          frame_no += 1

          if(frame_no % 4 != 0):
              continue

          process_frame = cv2.resize(frame, (150, 150))
          process_frame = np.expand_dims(process_frame, axis = 0)
          predictions = model.predict(process_frame, verbose = 0)
          pred_class = predictions[0]
          last_predictions.append(pred_class)
          
          if(len(last_predictions) > 3):
              last_predictions.pop(0)
          
          avg_prediction = sum(last_predictions) / len(last_predictions)

          if (avg_prediction < 0.15 and not last_result):
              last_result = True
              time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
              print("Start =", time)
              begin_time.append(time)

          if(avg_prediction > 0.15 and last_result):
              last_result = False
              time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
              print("End =", time)
              begin_time.append(time)

          cv2.imshow('frame', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break
      else:
          cv2.destroyAllWindows()
          break
