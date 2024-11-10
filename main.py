from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model2 = pickle.load(open('best_rf_model.pkl', 'rb'))
model1 = pickle.load(open('best_model.pkl', 'rb'))


def Find_disorder(answers):
    
    def preprocess_new_data(new_data):
         yes_no_columns = [
        'feeling_nervous', 'panic', 'breathing_rapidly', 'sweating', 'trouble_in_concentration',
        'having_trouble_in_sleeping', 'having_trouble_with_work', 'hopelessness', 'anger', 'over_react',
        'change_in_eating', 'suicidal_thought', 'feeling_tired', 'close_friend', 'social_media_addiction',
        'weight_gain', 'introvert', 'popping_up_stressful_memory', 'having_nightmares', 
        'avoids_people_or_activities', 'feeling_negative', 'trouble_concentrating', 
        'blamming_yourself', 'hallucinations', 'repetitive_behaviour', 'seasonally', 'increased_energy']

    
    
         for column in yes_no_columns:
                  new_data[column] = new_data[column].replace({'YES ': 'YES', ' NO ': 'NO', 'NO ': 'NO', 'YES': 'YES', 'NO': 'NO'})
                  new_data[column] = new_data[column].map({'YES': 1, 'NO': 0})

         return new_data
  
   
    
    
    def predict_new_data(answers):
        columns = ['feeling_nervous', 'panic', 'breathing_rapidly', 'sweating',
               'trouble_in_concentration', 'having_trouble_in_sleeping',
               'having_trouble_with_work', 'hopelessness', 'anger', 'over_react',
               'change_in_eating', 'suicidal_thought', 'feeling_tired', 'close_friend',
               'social_media_addiction', 'weight_gain', 'introvert',
               'popping_up_stressful_memory', 'having_nightmares',
               'avoids_people_or_activities', 'feeling_negative',
               'trouble_concentrating', 'blamming_yourself', 'hallucinations',
               'repetitive_behaviour', 'seasonally', 'increased_energy']
        new_data = pd.DataFrame([answers], columns=columns)

        new_data_processed = preprocess_new_data(new_data)
    
  
        new_data_processed = new_data_processed.reindex(columns=columns, fill_value=0)
    
    
        predictions = model1.predict(new_data_processed)
    
        return predictions
    

    predictions = predict_new_data(answers)
    
    def decode_result(predictions):
        if predictions== 0:
            return 'ADHD'
        elif predictions== 1:
            return 'ASD'
        elif predictions== 2:
            return 'LONELINESS'
        elif predictions== 3:
            return 'MDD'
        elif predictions== 4:
            return 'OCD'
        elif predictions== 5:
            return 'PDD'
        elif predictions== 6:
            return 'PTSD'
        elif predictions== 7:
            return 'ANEXITY'
        elif predictions== 8:
            return 'BiPolar'
        elif predictions== 9:
            return 'Eating Disorder'
        elif predictions== 10:
            return 'Psychotic depression'
        elif predictions== 11:
            return 'sleeping disorder'
        
    result=decode_result(predictions)
    return result

def find_disorder(raw_values):
    def encode_input(values):
        encoded_values = []
        yes_no_encoder = {'YES': 1, 'NO': 0}
        frequency_encoder = {'Usually': 3, 'Most-Often': 2, 'Sometimes': 1, 'Seldom': 0}
        top_features=['Mood Swing', 'Sadness', 'Optimisim', 'Sexual Activity', 'Euphoric', 'Suicidal thoughts', 'Exhausted', 'Concentration', 'Aggressive Response', 'Nervous Break-down']
        for feature, value in zip(top_features, values):
            if feature in ['Mood Swing', 'Optimisim', 'Sadness', 'Sexual Activity', 
                       'Euphoric', 'Nervous Break-down', 'Exhausted', 'Suicidal thoughts', 
                       'Concentration', 'Overthinking']:
                if isinstance(value, str) and value in yes_no_encoder:
                    encoded_values.append(yes_no_encoder[value])
                elif feature in ['Concentration', 'Optimisim']:
                    encoded_values.append(int(value))
                else:
                
                    encoded_values.append(frequency_encoder.get(value, 0))
            else:
            
                    encoded_values.append(0)
    
        return np.array([encoded_values])
    
   
    
    
    
    def predict_disorder(raw_values):
            processed_data = encode_input(raw_values)
            prediction = model2.predict(processed_data)
            if prediction== 0:
                 return 'Bipolar Type-1'
            elif prediction== 1:
                 return 'Bipolar Type-2'
            elif prediction== 2:
                return 'Depression'
            elif prediction== 3:
                return 'Normal'

    prediction=predict_disorder(raw_values)
    return prediction

@app.route('/generalPredict', methods=['POST'])
def model1Api():
    data = request.get_json()
    listOfDics = data["data"]
    answers = []
    counter = 0
    features = 27
    for dictionary in listOfDics:
        if "answer" in dictionary:
            answers.append(dictionary["answer"])
            counter += 1
    if counter != features:
        return jsonify("You didn't answer all the questions! Try again.")
    answers = np.array(answers)
    prediction = Find_disorder(answers)  # make the prediction
    return jsonify({"Based on your test results, you have been diagnosed with a mental health condition identified as: \n":prediction})



@app.route('/depressionPredict', methods=['POST'])
def model2Api():
    data = request.get_json()
    dicsList = data['data']
    answers = []
    num_of_ans = 0
    features = 10
    for dictionary in dicsList:
        if 'answer' in dictionary:
            answers.append(dictionary['answer'])
            num_of_ans += 1

    if num_of_ans != features:
        return jsonify("You didn't answer all the questions! Try again.")
    answers = np.array(answers)
    result = find_disorder(answers)
    return jsonify({"Based on your test results, you have been diagnosed with a mental health condition identified as: \n": result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
