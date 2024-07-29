from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from tools import converse
import json


def index(request: HttpRequest) -> HttpResponse:
    return render(request, 'index.html')


@csrf_exempt
def ask(request: HttpRequest) -> JsonResponse:
    params = json.loads(request.body)
    question = params['q']
    history = []

    for exchange in params['h']:
        history.append(('human', exchange['question']))
        history.append(('ai', exchange['answer']))

    return JsonResponse({
        'data': converse.run(
            question,
            history,
        ),
        'message': 'Success',
        'success': True,
    })
