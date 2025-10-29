/**
 * MUI theme configuration for PhonoLex
 *
 * Minimalist, sophisticated design with restrained color palette
 * Warm neutrals with deep teal and terracotta accents
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';

const getDesignTokens = (mode: 'light' | 'dark'): ThemeOptions => ({
  palette: {
    mode,
    ...(mode === 'light'
      ? {
          // Light mode - Sophisticated minimalism
          primary: {
            main: '#1E5A5A', // Deep cypress green/teal
            light: '#2D7373',
            dark: '#164545',
            contrastText: '#FAFAF8',
          },
          secondary: {
            main: '#C4675F', // Warm terracotta/clay
            light: '#D4827C',
            dark: '#A85149',
            contrastText: '#FAFAF8',
          },
          background: {
            default: '#FAFAF8', // Warm off-white
            paper: '#FFFFFF',
          },
          text: {
            primary: '#2C2C2C', // Charcoal (softer than pure black)
            secondary: '#5A5A5A', // Medium gray
          },
          error: {
            main: '#B85450', // Muted red (less alarming)
            light: '#C77572',
            dark: '#9A4340',
          },
          warning: {
            main: '#C4895F', // Warm amber (less bright)
            light: '#D4A47C',
            dark: '#A87249',
          },
          info: {
            main: '#4A7C8C', // Muted blue-gray
            light: '#6B96A3',
            dark: '#3A6270',
          },
          success: {
            main: '#5A7C5A', // Muted sage green
            light: '#7A967A',
            dark: '#4A6349',
          },
          divider: 'rgba(44, 44, 44, 0.08)', // Very subtle
        }
      : {
          // Dark mode - Warm charcoal with muted accents
          primary: {
            main: '#5A9D9D', // Lighter teal for dark mode
            light: '#7AB5B5',
            dark: '#4A8585',
            contrastText: '#2C2C2C',
          },
          secondary: {
            main: '#D4827C', // Lighter terracotta for dark mode
            light: '#E0A39E',
            dark: '#C4675F',
            contrastText: '#2C2C2C',
          },
          background: {
            default: '#1A1A18', // Warm dark charcoal
            paper: '#252523',
          },
          text: {
            primary: '#E8E8E6', // Warm off-white
            secondary: '#A8A8A6', // Medium gray
          },
          error: {
            main: '#C77572',
            light: '#D49694',
            dark: '#B85450',
          },
          warning: {
            main: '#D4A47C',
            light: '#E0BA9E',
            dark: '#C4895F',
          },
          info: {
            main: '#6B96A3',
            light: '#8BADB8',
            dark: '#4A7C8C',
          },
          success: {
            main: '#7A967A',
            light: '#96AD96',
            dark: '#5A7C5A',
          },
          divider: 'rgba(232, 232, 230, 0.08)',
        }),
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  spacing: 8,
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          padding: '10px 20px',
          fontWeight: 500,
          letterSpacing: '0.01em',
          '&:focus-visible': {
            outline: '2px solid currentColor',
            outlineOffset: '2px',
          },
        },
        sizeLarge: {
          padding: '14px 28px',
          fontSize: '1rem',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          },
        },
      },
      defaultProps: {
        disableElevation: true,
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          border: '1px solid',
          borderColor: 'rgba(44, 44, 44, 0.06)',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
          '&:focus-visible': {
            outline: '2px solid currentColor',
            outlineOffset: '2px',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
        elevation1: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
        },
        elevation2: {
          boxShadow: '0 2px 6px rgba(0, 0, 0, 0.06)',
        },
        elevation3: {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: 'outlined',
      },
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 6,
            '& fieldset': {
              borderColor: 'rgba(44, 44, 44, 0.12)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(44, 44, 44, 0.24)',
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
          borderRadius: 4,
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          border: '1px solid',
          borderColor: 'rgba(44, 44, 44, 0.06)',
          boxShadow: 'none',
          '&:before': {
            display: 'none',
          },
          '&.Mui-expanded': {
            margin: 0,
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          '&:focus-visible': {
            outline: '2px solid currentColor',
            outlineOffset: '-2px',
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          '&:focus-visible': {
            outline: '2px solid currentColor',
            outlineOffset: '2px',
          },
        },
      },
    },
  },
});

export const lightTheme = createTheme(getDesignTokens('light'));
export const darkTheme = createTheme(getDesignTokens('dark'));

// Export default as light theme (professional standard)
export const theme = lightTheme;
