import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, Button, Image, ActivityIndicator, Dimensions } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native'; // Necessário para Expo

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isModelReady, setIsModelReady] = useState(false);
  const [cameraRef, setCameraRef] = useState<any>(null);  // O tipo da ref pode ser qualquer tipo
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isProcessing, setIsProcessing] = useState(false); // Para mostrar o indicador de carregamento durante a análise

  const camera = useRef(null); // Melhor usar o useRef para referência da câmera

  const screenWidth = Dimensions.get('window').width;

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();

    const loadModel = async () => {
      await tf.ready();
      const model = await cocoSsd.load();
      setModel(model);
      setIsModelReady(true);
    };

    loadModel();
  }, []);

  const takePhoto = async () => {
    if (cameraRef) {
      const photo = await cameraRef.takePictureAsync();
      setPhotoUri(photo.uri);
      analyzePhoto(photo.uri);
    }
  };

  const analyzePhoto = async (uri: string) => {
    setIsProcessing(true);

    // Carregar a imagem como um tensor para análise
    const response = await fetch(uri);
    const imageBlob = await response.blob();
    const imageBitmap = await createImageBitmap(imageBlob);

    // Preparar a imagem para o TensorFlow
    const imageTensor = tf.browser.fromPixels(imageBitmap);
    const predictions = await model.detect(imageTensor);

    setPredictions(predictions);
    setIsProcessing(false);
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
          <Image source={{ uri: photoUri }} style={[styles.preview, { width: screenWidth - 40, height: (screenWidth - 40) * (3 / 4) }]} />
          <Text>Resultados:</Text>
          {isProcessing ? (
            <ActivityIndicator size="large" color="#0000ff" />
          ) : (
            predictions.length > 0 ? (
              predictions.map((p, index) => (
                <Text key={index}>
                  {p.class}: {(p.score * 100).toFixed(2)}%
                </Text>
              ))
            ) : (
              <Text>Sem objetos detectados.</Text>
            )
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
    justifyContent: 'flex-end',
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
    borderRadius: 10,
    marginBottom: 10,
  },
});
