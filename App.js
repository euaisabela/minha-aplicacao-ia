import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Button, Image, ActivityIndicator } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native'; // Necessário para Expo

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [isModelReady, setIsModelReady] = useState(false);
  const [cameraRef, setCameraRef] = useState(null);
  const [photoUri, setPhotoUri] = useState(null);
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();

    // Carregar modelo COCO-SSD
    const loadModel = async () => {
      await tf.ready();
      const model = await cocoSsd.load();
      setModel(model);
      setIsModelReady(true);
    };

    loadModel();
  }, []);

  const [model, setModel] = useState(null);

  const takePhoto = async () => {
    if (cameraRef) {
      const photo = await cameraRef.takePictureAsync();
      setPhotoUri(photo.uri);
      analyzePhoto(photo.uri);
    }
  };

  const analyzePhoto = async (uri) => {
    const response = await fetch(uri);
    const blob = await response.blob();
    const image = new Image();
    image.src = URL.createObjectURL(blob);

    const predictions = await model.detect(image);
    setPredictions(predictions);
  };

  if (hasPermission === null) {
    return <Text>Solicitando permissão para acessar a câmera...</Text>;
  }

  if (hasPermission === false) {
    return <Text>Sem acesso à câmera.</Text>;
  }

  return (
    <View style={styles.container}>
      {!isModelReady ? (
        <View style={styles.loader}>
          <Text>Carregando modelo...</Text>
          <ActivityIndicator size="large" color="#0000ff" />
        </View>
      ) : (
        <Camera style={styles.camera} ref={(ref) => setCameraRef(ref)}>
          <View style={styles.controls}>
            <Button title="Tirar Foto" onPress={takePhoto} />
          </View>
        </Camera>
      )}
      {photoUri && (
        <View style={styles.result}>
          <Image source={{ uri: photoUri }} style={styles.preview} />
          <Text>Resultados:</Text>
          {predictions.length > 0 ? (
            predictions.map((p, index) => (
              <Text key={index}>
                {p.class}: {(p.score * 100).toFixed(2)}%
              </Text>
            ))
          ) : (
            <Text>Processando...</Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  controls: {
    position: 'absolute',
    bottom: 20,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loader: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  result: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  preview: {
    width: 300,
    height: 300,
  },
});
