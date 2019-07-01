def prepareInputOutput():
  trainingInputFrames=[] # list of list (LSTM_TIME_STEPS) of 20*3=60 data points
  trainingOutputFrames=[] # list of 20*3=60 data points
  testInputFrames=[]
  testOutputFrames=[]
  for action_no in range(NUMBER_OF_ACTIONS) :
   for subject_no in range(NUMBER_OF_SUBJECTS_PER_ACTION) :
     for example_no in range(MAX_NUMBER_OF_EXAMPLES_PER_SUBJECT) :
        if action_no in data_dictionary and subject_no in data_dictionary[action_no] and  example_no in data_dictionary[action_no][subject_no] :
          inputFrames=trainingInputFrames
          outputFrames=trainingOutputFrames
          if action_no == TEST_ACTION_NO and subject_no == TEST_SUBJECT_NO and example_no == TEST_EXAMPLE_NO :
           inputFrames=testInputFrames
           outputFrames=testOutputFrames
                     
          for i in range(LSTM_TIME_STEPS,len(normalized_data_dictionary[action_no][subject_no][example_no])) :
              input_frame_time_steps=normalized_data_dictionary[action_no][subject_no][example_no][i-LSTM_TIME_STEPS:i]
              #print(input_frame_time_steps[0].shape)
              reshape_list_20x3_to_60(input_frame_time_steps)
              #print(input_frame_time_steps[0].shape)
              inputFrames.append(input_frame_time_steps)
              output_frame=normalized_data_dictionary[action_no][subject_no][example_no][i]
              output_frame=reshape_20x3_to_60(output_frame) ## convert 20,3  to 60
              #print(output_frame.shape)
              outputFrames.append(output_frame)
  return np.array(trainingInputFrames),np.array(trainingOutputFrames),np.array(testInputFrames),np.array(testOutputFrames)         


load_data()
centralize_data_taking_point_7_as_center()
normalize_data()
model=buildModel()
trainingInputFrames,trainingOutputFrames,testInputFrames,testOutputFrames=prepareInputOutput()
model.fit(trainingInputFrames, trainingOutputFrames, epochs = 10, batch_size = 1,verbose=2)
predictions=model.predict(testInputFrames)
mean_squared_error=(np.square(testOutputFrames-predictions)).mean(axis=0).mean(axis=0)
predictions=denormalizeFrames(reshape_list_60_to_20x3(predictions.tolist()))
animate(predictions,"predicitons for action=10,subject=5,example=2")

