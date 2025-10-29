/**
 * Expandable Tool Card Component
 *
 * A collapsible card that displays a tool with:
 * - Header with icon, title, and description
 * - Expandable content area
 * - Smooth animations
 */

import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Collapse,
  IconButton,
  Typography,
  Box,
  Avatar,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material';

interface ExpandableToolCardProps {
  icon: React.ReactElement;
  title: string;
  description: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  color?: string;
}

const ExpandableToolCard: React.FC<ExpandableToolCardProps> = ({
  icon,
  title,
  description,
  children,
  defaultExpanded = false,
  color = 'primary.main',
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  const handleExpandClick = () => {
    setExpanded(!expanded);
  };

  return (
    <Card
      elevation={expanded ? 3 : 1}
      sx={{
        mb: 2,
        transition: 'all 0.3s ease',
        '&:hover': {
          elevation: 2,
          boxShadow: 2,
        },
      }}
    >
      <CardHeader
        avatar={
          <Avatar sx={{ bgcolor: color }}>
            {icon}
          </Avatar>
        }
        action={
          <IconButton
            onClick={handleExpandClick}
            aria-expanded={expanded}
            aria-label="show more"
            sx={{
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s',
            }}
          >
            <ExpandMoreIcon />
          </IconButton>
        }
        title={
          <Typography variant="h6" component="div" fontWeight={600}>
            {title}
          </Typography>
        }
        subheader={
          <Typography variant="body2" color="text.secondary">
            {description}
          </Typography>
        }
        sx={{
          cursor: 'pointer',
          '&:hover': {
            bgcolor: 'action.hover',
          },
        }}
        onClick={handleExpandClick}
      />
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <CardContent>
          <Box sx={{ pt: 1 }}>
            {children}
          </Box>
        </CardContent>
      </Collapse>
    </Card>
  );
};

export default ExpandableToolCard;
