import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import 'dart:io';

import 'package:image_picker/image_picker.dart';

import 'package:http/http.dart' as http;
import 'package:firebase_storage/firebase_storage.dart';
import 'dart:convert';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    return MaterialApp(

      theme: ThemeData.dark(),
      home:MyHomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    // TODO: implement createState
    return _MyHomePage();
  }
}




class _MyHomePage extends State<MyHomePage> {

  File image;

  Future getCameraImage() async {
    var img = await ImagePicker.pickImage(source: ImageSource.camera);

    setState(() {
      image = img;
    });
  }

  Future getGalleryImage() async{
    var img = await ImagePicker.pickImage(source: ImageSource.gallery);

    setState(() {
      image = img;
    });
  }



  void upload() async {


    await FirebaseStorage.instance.ref().child('input.jpg').putFile(image);

    Navigator.push(context, MaterialPageRoute(builder: (BuildContext context)=>Prediction()));


  }

  @override
  Widget build(BuildContext context) {
    // TODO: implement build


    return Scaffold(
      appBar: AppBar(title: Text('Mango Prediction'),),
      body: Container(
        child: Column(
          children: <Widget>[
            SizedBox(
              height: 15.0,
            ),
            Expanded(
              child: Container(
                child: Center(
                  child: image == null ? Text('No Image Selected'): Image.file(image),
                ),
              ),
              flex: 2,
            ),
            Expanded(
              child: Center(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    FlatButton(
                      onPressed: getCameraImage,
                      child: Text('Capture from Camera'),
                      color: Colors.green,
                    ),
                    FlatButton(
                      onPressed: getGalleryImage,
                      child: Text('Select from Gallery'),
                      color: Colors.green,
                    )
                  ],
                ),
              ),
            ),
            Expanded(
              child: Center(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: <Widget>[
                    RaisedButton(
                      child: Text('Upload'),
                      color:Colors.orange,

                      onPressed: () async{
                        await upload();
                        showDialog(context: context,
                            builder: (BuildContext context){

                              return AlertDialog(title: Center(
                                child: Column(
                                  children: <Widget>[
                                    CircleAvatar(child:
                                      Icon(Icons.assignment_turned_in,
                                        color: Colors.green,size: 60,
                                      ),
                                      radius: 40,
                                    ),
                                    SizedBox(height: 15,),
                                    Text('Upload Completed',
                                      style: TextStyle(
                                        fontWeight: FontWeight.bold,
                                        color: Colors.black
                                      )
                                    )
                                  ],
                                ),
                              ),
                                backgroundColor: Colors.white,

                              );

                            }

                        );

                      },
                    ),
                    SizedBox(
                      height: 20,
                    ),
                  ],
                ),
              ),
              flex: 2,
            )
          ],
        ),
      ),
    );
  }
}

class Prediction extends StatefulWidget{
  @override
  State<StatefulWidget> createState() {
    // TODO: implement createState
    return _Prediction();
  }
}

class _Prediction extends State<Prediction>{


  Widget getResult(){
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[

        ],
      );
  }

  var data;
  var count='0';
   var url='https://user-images.githubusercontent.com/194400/49531010-48dad180-f8b1-11e8-8d89-1e61320e1d82.png';
  Future getData() async{
    http.Response  response= await http.get("http://192.168.1.102:107");

    return response.body;
  }


  @override
  Widget build(BuildContext context) {
    // TODO: implement build

    return Scaffold(
      appBar: AppBar(title: Text('Your Predictions'),),
      body: Container(
        child: Center(
            child:Column(
              children: <Widget>[
                FlatButton(
                  onPressed: ()async {
                    data = await getData();
                    setState(() {
                      url = jsonDecode(data)['url'];
                      count = jsonDecode(data)['count'];
                    });
                    print(url);
                    print(count);
                  },
                  child: Text('Predict'),
                  color: Colors.purple,
                ),
                SizedBox(height: 20,),
                Expanded(
                  child: Text('Number of Mangos Detected : ' + count,style: TextStyle(fontSize: 30.0,fontWeight: FontWeight.bold),),
                ),
                SizedBox(height: 10,),
                Expanded(
                    child: Image.network(url,height: 200,width: 300,),
                ),

                SizedBox(height: 20,),
                FlatButton(
                  child: Text('Back to Home',style: TextStyle(fontSize: 20.0),),
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  color: Colors.purple,
                )
              ],
            )

        )
      ),

    );
  }

}