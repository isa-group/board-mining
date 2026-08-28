/**
 * Board Design Moddle Types
 *
 * Defines the diagram-js element types for board design visualization:
 * - BoardDesign: root container
 * - ListBox: individual board list
 * - CardFlow: connected component container
 * - SemanticPreference: transition between lists
 */

export const boardTypes = {
  BoardDesign: {
    isAbstract: true,
    properties: [
      { name: 'lists', type: 'ListBox', isMany: true },
      { name: 'semanticPreferences', type: 'SemanticPreference', isMany: true },
      { name: 'cardFlows', type: 'CardFlow', isMany: true },
    ],
  },

  ListBox: {
    superClass: ['Shape'],
    properties: [
      { name: 'name', type: 'String' },
      { name: 'listId', type: 'String' },
      { name: 'isCreateList', type: 'Boolean', default: false },
      { name: 'isCloseList', type: 'Boolean', default: false },
      { name: 'isUseList', type: 'Boolean', default: false },
      { name: 'trafficVolume', type: 'Number', default: 0 },
    ],
  },

  SemanticPreference: {
    superClass: ['Connection'],
    properties: [
      { name: 'sourceList', type: 'String' }, // listId
      { name: 'targetList', type: 'String' }, // listId
      { name: 'volume', type: 'Number', default: 0 },
    ],
  },

  CardFlow: {
    superClass: ['Container'],
    properties: [
      { name: 'lists', type: 'ListBox', isMany: true },
      { name: 'componentId', type: 'String' },
    ],
  },
}
