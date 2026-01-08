// src/App.test.js
import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

test('renders Deepfake Audio Detector title', () => {
  render(<App />);
  expect(screen.getByText(/Deepfake Audio Detector/i)).toBeInTheDocument();
});

test('shows start recording button initially', () => {
  render(<App />);
  expect(screen.getByText(/START RECORDING/i)).toBeInTheDocument();
});